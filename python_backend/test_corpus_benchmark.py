"""Benchmark retrieval models on the independent documents in the project test folder."""

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

import search_utils


ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT_DIR / "test"
DATASET_FILE = TEST_DIR / "retrieval_benchmark_dataset.json"
MANUAL_DATASET_FILE = TEST_DIR / "retrieval_manual_queries.json"
REPORT_FILE = TEST_DIR / "retrieval_benchmark_report.json"
SAMPLES_PER_DOCUMENT = 12
TOP_K_VALUES = (5, 10, 30)
# A normal application search asks for 5 results and blends up to 40 candidates
# (max(n_results * 8, 40), capped at 250).
RETRIEVAL_DEPTH = 40
CURRENT_EMBEDDING = "BAAI/bge-m3"
ALTERNATIVE_EMBEDDING = "intfloat/multilingual-e5-large"
LEGACY_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"
ALTERNATIVE_RERANKER = "BAAI/bge-reranker-v2-m3"
PRODUCTION_MODE = "Production: BGE-M3 + BM25 (weighted fusion)"
LEGACY_MINILM_MODE = "Legacy: BGE-M3 + BM25 + MiniLM"
RANDOM_SEED = 20260928


def _extract_docx(path: Path) -> str:
    document = Document(str(path))
    blocks = []
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = Paragraph(child, document).text.strip()
            if text:
                blocks.append(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = []
                seen_cells = set()
                for cell in row.cells:
                    cell_key = id(cell._tc)
                    if cell_key in seen_cells:
                        continue
                    seen_cells.add(cell_key)
                    cells.append(" ".join(cell.text.split()))
                rows.append("\t".join(cells))
            table_text = "\n".join(row for row in rows if row.strip("\t "))
            if table_text:
                blocks.append(table_text)
    return "\n\n".join(blocks)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"(?u)\w+", text.lower()))


def _similarity(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _extract_test_documents() -> tuple[list[dict[str, str]], list[str]]:
    candidates = []
    for path in sorted(TEST_DIR.iterdir(), key=lambda item: (item.suffix != ".md", item.name.lower())):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".md":
            text = path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            text = _extract_docx(path)
        elif suffix == ".pdf":
            text = _extract_pdf(path)
        else:
            continue
        text = text.strip()
        if len(text) >= 100:
            candidates.append({"name": path.name, "text": text, "format": suffix[1:]})

    unique_documents = []
    duplicates = []
    for candidate in candidates:
        duplicate_of = next(
            (
                existing["name"]
                for existing in unique_documents
                if _similarity(candidate["text"], existing["text"]) >= 0.90
            ),
            None,
        )
        if duplicate_of:
            duplicates.append(f"{candidate['name']} duplicates {duplicate_of}; excluded from scoring")
        else:
            unique_documents.append(candidate)
    return unique_documents, duplicates


def _is_header(block: str) -> bool:
    lines = block.splitlines()
    return any(
        line.strip().startswith("#")
        or re.fullmatch(r"\s*(?:=+|-{2,})\s*", line)
        for line in lines
    )


def _segment_text(text: str, source_id: str) -> list[dict[str, str]]:
    """Mirror src/utils/segmentation.js so benchmark gold IDs match the app."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] == "\n":
            pos += 1
        if pos >= len(text):
            break
        start = pos
        last_content_end = pos
        while pos < len(text):
            if text[pos] == "\n" and pos + 1 < len(text) and text[pos + 1] == "\n":
                break
            if text[pos] != "\n":
                last_content_end = pos + 1
            pos += 1
        end = last_content_end
        value = text[start:end]
        if value.strip():
            blocks.append({"start": start, "end": end, "header": _is_header(value)})

    grouped: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    def finish_current() -> None:
        nonlocal current
        if current:
            grouped.append(current)
            current = None

    for block in blocks:
        if current is None:
            current = {"start": block["start"], "end": block["end"], "blocks": [block]}
        elif block["header"]:
            if all(part["header"] for part in current["blocks"]):
                current["blocks"].append(block)
                current["end"] = block["end"]
            else:
                finish_current()
                current = {"start": block["start"], "end": block["end"], "blocks": [block]}
        elif all(part["header"] for part in current["blocks"]) or current["end"] - current["start"] < 220:
            current["blocks"].append(block)
            current["end"] = block["end"]
        else:
            finish_current()
            current = {"start": block["start"], "end": block["end"], "blocks": [block]}
    finish_current()

    for index in range(len(grouped) - 1, 0, -1):
        if grouped[index]["end"] - grouped[index]["start"] < 100:
            grouped[index - 1]["end"] = grouped[index]["end"]
            grouped[index - 1]["blocks"].extend(grouped[index]["blocks"])
            grouped.pop(index)
    if len(grouped) > 1 and grouped[0]["end"] - grouped[0]["start"] < 100:
        grouped[1]["start"] = grouped[0]["start"]
        grouped[1]["blocks"] = grouped[0]["blocks"] + grouped[1]["blocks"]
        grouped.pop(0)

    return [{
        "id": f"{source_id}::seg_{index}",
        "source": source_id,
        "text": text[segment["start"]:segment["end"]],
    } for index, segment in enumerate(grouped)]


def _load_corpus() -> tuple[list[dict[str, str]], list[str], str]:
    documents, duplicates = _extract_test_documents()
    corpus = []
    fingerprint = hashlib.sha256()
    for document in documents:
        source_id = re.sub(r"[^\w.-]+", "_", document["name"], flags=re.UNICODE)
        fingerprint.update(document["name"].encode("utf-8"))
        fingerprint.update(document["text"].encode("utf-8"))
        corpus.extend(_segment_text(document["text"], source_id))
    return corpus, duplicates, fingerprint.hexdigest()


def _load_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY in the environment to generate query cases; "
            "no key is embedded in source code."
        )
    from google import genai

    return genai.Client(api_key=api_key)


def _generate_dataset(corpus: list[dict[str, str]], fingerprint: str) -> list[dict[str, str]]:
    client = _load_gemini_client()
    by_source: dict[str, list[dict[str, str]]] = {}
    for segment in corpus:
        if len(segment["text"].strip()) >= 80:
            by_source.setdefault(segment["source"], []).append(segment)

    rng = random.Random(RANDOM_SEED)
    selected = []
    for source, segments in sorted(by_source.items()):
        sample = rng.sample(segments, min(SAMPLES_PER_DOCUMENT, len(segments)))
        selected.extend((source, segment) for segment in sample)

    dataset = []
    print(f"Generating Gemini queries: {len(selected)} target segments across {len(by_source)} documents")
    for source, segment in tqdm(selected):
        prompt = (
            "Create one realistic information-retrieval query that a reader might type to find the passage. "
            "Use the passage's language, 3-9 words, and do not copy a full sentence. "
            "Return only the query, with no quotes or explanation.\n\n"
            f"Passage:\n{segment['text'][:1800]}\n\nQuery:"
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            query = re.sub(r"\s+", " ", (response.text or "").strip().strip('"'))
            if len(query.split()) < 2:
                continue
            dataset.append({
                "query": query,
                "expected_id": segment["id"],
                "source": source,
                "kind": "gemini",
                "text_preview": segment["text"][:120].replace("\n", " "),
            })
        except Exception as error:
            print(f"\nGemini query generation failed for {source}/{segment['id']}: {error}")

    DATASET_FILE.write_text(
        json.dumps({"fingerprint": fingerprint, "cases": dataset}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset


def _load_or_generate_dataset(corpus: list[dict[str, str]], fingerprint: str, regenerate: bool, manual_only: bool = False):
    if manual_only:
        return []
    saved_cases = []
    if not regenerate and DATASET_FILE.exists():
        try:
            saved = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint and saved.get("cases"):
                saved_cases = saved["cases"]
                print(f"Using saved Gemini/legacy test-corpus dataset with {len(saved_cases)} queries")
        except (json.JSONDecodeError, AttributeError):
            pass
    if regenerate:
        saved_cases = _generate_dataset(corpus, fingerprint)
    for index, case in enumerate(saved_cases):
        case.setdefault("kind", "gemini")
        case.setdefault("case_id", f"gemini-{index + 1:03d}")
    return saved_cases


def _load_manual_cases(corpus: list[dict[str, str]], fingerprint: str) -> list[dict[str, Any]]:
    if not MANUAL_DATASET_FILE.exists():
        return []
    raw = json.loads(MANUAL_DATASET_FILE.read_text(encoding="utf-8"))
    cases = raw.get("cases", []) if isinstance(raw, dict) else raw
    available_ids = {segment["id"] for segment in corpus}
    valid = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not str(case.get("query", "")).strip():
            print(f"Skipping manual case {index + 1}: missing query")
            continue
        relevant_ids = _gold_ids(case)
        missing = relevant_ids - available_ids
        if not relevant_ids or missing:
            print(f"Skipping manual case {index + 1}: invalid segment IDs: {sorted(missing)}")
            continue
        case = dict(case)
        case["query"] = re.sub(r"\s+", " ", str(case["query"])).strip()
        case["relevant_ids"] = sorted(relevant_ids)
        case["source"] = str(case.get("source") or "manual")
        case["kind"] = "manual"
        case.setdefault("case_id", f"manual-{index + 1:03d}")
        valid.append(case)
    saved_fingerprint = raw.get("fingerprint") if isinstance(raw, dict) else None
    print(f"Loaded {len(valid)} manually labeled queries" + (" (corpus fingerprint differs; verify IDs)" if saved_fingerprint and saved_fingerprint != fingerprint else ""))
    return valid


def _encode_documents(model: SentenceTransformer, texts: list[str], prefix: str = "") -> np.ndarray:
    model_inputs = [f"{prefix}{text}" for text in texts]
    return np.asarray(
        model.encode(
            model_inputs,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def _query_input(query: str, model_key: str) -> str:
    if model_key == "e5":
        return f"query: {query}"
    return query


def _hybrid_candidates(query, semantic_scores, bm25_scores, segments, retrieval_depth):
    semantic_order = np.argsort(semantic_scores)[::-1][:retrieval_depth]
    lexical_order = [
        int(index)
        for index in np.argsort(bm25_scores)[::-1]
        if bm25_scores[index] > 0
    ][:retrieval_depth]
    candidate_indices = sorted(set(semantic_order.tolist()) | set(lexical_order))
    if not candidate_indices:
        return []

    semantic_max = max((float(semantic_scores[index]) for index in semantic_order), default=0.0)
    lexical_max = max((float(bm25_scores[index]) for index in lexical_order), default=0.0)
    candidates = []
    for index in candidate_indices:
        text = segments[index]["text"]
        overlap = search_utils.keyword_overlap_score(query, text)
        score = (
            0.35 * (float(semantic_scores[index]) / semantic_max if semantic_max > 0 else 0.0)
            + 0.35 * (float(bm25_scores[index]) / lexical_max if lexical_max > 0 else 0.0)
            + 0.30 * overlap
        )
        candidates.append((index, score, overlap))
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    # Production weighted_hybrid_search returns at most top_k_retrieval
    # candidates before reranking; keep the benchmark's pool identical.
    return candidates[:retrieval_depth]


def _rerank(query, candidates, segments, reranker: CrossEncoder):
    if not candidates:
        return []
    pairs = [[query, segments[index]["text"]] for index, _, _ in candidates]
    cross_scores = np.asarray(reranker.predict(pairs, batch_size=16, show_progress_bar=False))
    ranked = []
    for (index, hybrid_score, overlap), cross_score in zip(candidates, cross_scores):
        probability = 1.0 / (1.0 + np.exp(-np.clip(float(cross_score), -60, 60)))
        score = 0.55 * probability + 0.35 * overlap + 0.10 * hybrid_score
        ranked.append((segments[index]["id"], score))
    ranked.sort(key=lambda result: result[1], reverse=True)
    return [segment_id for segment_id, _ in ranked]


def _bootstrap_interval(values, seed=RANDOM_SEED, repetitions=3000, clusters=None):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    if clusters and len(clusters) == len(values):
        groups: dict[str, list[float]] = {}
        for cluster, value in zip(clusters, values):
            groups.setdefault(str(cluster), []).append(float(value))
        grouped_values = [np.asarray(group, dtype=np.float64) for group in groups.values()]
        draws = rng.integers(0, len(grouped_values), size=(repetitions, len(grouped_values)))
        samples = np.asarray([
            np.concatenate([grouped_values[index] for index in row]).mean()
            for row in draws
        ])
    else:
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _gold_ids(case: dict[str, Any]) -> set[str]:
    values = case.get("relevant_ids", case.get("expected_ids"))
    if values is None:
        values = [case["expected_id"]] if case.get("expected_id") else []
    elif isinstance(values, str):
        values = [values]
    return {str(value) for value in values}


def _rank_metrics(ordered_ids: list[str], relevant_ids: set[str], top_k: int) -> tuple[float, float, float]:
    if not relevant_ids:
        return 0.0, 0.0, 0.0
    top = ordered_ids[:top_k]
    relevant_ranks = [index + 1 for index, value in enumerate(top) if value in relevant_ids]
    hit = float(bool(relevant_ranks))
    reciprocal_rank = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
    recall = len(relevant_ranks) / len(relevant_ids)
    return hit, reciprocal_rank, recall


def _case_cluster(case: dict[str, Any]) -> str:
    if case.get("cluster_id"):
        return str(case["cluster_id"])
    return "|".join(sorted(_gold_ids(case)))


def _evaluate_rankings(rankings, cases):
    top_ks = TOP_K_VALUES
    print("\n" + "=" * 112)
    print(f"TEST CORPUS MODEL COMPARISON | {len(cases)} queries | {len({case['source'] for case in cases})} sources")
    print("Multiple relevant segments supported; 95% bootstrap interval for Hit@5")
    print("=" * 112)
    print(f"{'Retrieval mode':<52} {'K':>3} {'Hit Rate':>11} {'MRR':>9} {'Recall':>9} {'Hit@5 95% CI':>23}")
    print("-" * 112)

    for mode, query_rankings in rankings.items():
        for top_k in top_ks:
            hits = []
            reciprocal_ranks = []
            recalls = []
            for case, ordered_ids in zip(cases, query_rankings):
                hit, reciprocal_rank, recall = _rank_metrics(ordered_ids, _gold_ids(case), top_k)
                hits.append(hit)
                reciprocal_ranks.append(reciprocal_rank)
                recalls.append(recall)
            hit_rate = float(np.mean(hits)) * 100 if hits else 0.0
            mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            ci = _bootstrap_interval(
                hits,
                clusters=[_case_cluster(case) for case in cases],
            ) if top_k == 5 else (0.0, 0.0)
            ci_label = f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]" if top_k == 5 else ""
            print(f"{mode:<52} {top_k:>3} {hit_rate:>10.1f}% {mrr:>9.4f} {float(np.mean(recalls)) if recalls else 0.0:>9.4f} {ci_label:>23}")
        print("-" * 132)

    baseline = PRODUCTION_MODE
    if baseline in rankings:
        print("\nPAIRED HIT@5 DIFFERENCE VS CURRENT PRODUCTION")
        print("Positive difference favors the compared mode; intervals resample relevant-passage clusters.")
        print(f"{'Compared mode':<52} {'Delta':>9} {'95% CI':>23} {'Win/Loss/Tie':>16}")
        print("-" * 108)
        base_hits = [
            _rank_metrics(ids, _gold_ids(case), 5)[0]
            for case, ids in zip(cases, rankings[baseline])
        ]
        clusters = [_case_cluster(case) for case in cases]
        for mode, query_rankings in rankings.items():
            if mode == baseline:
                continue
            candidate_hits = [
                _rank_metrics(ids, _gold_ids(case), 5)[0]
                for case, ids in zip(cases, query_rankings)
            ]
            deltas = [candidate - current for candidate, current in zip(candidate_hits, base_hits)]
            low, high = _bootstrap_interval(deltas, seed=RANDOM_SEED + 1, clusters=clusters)
            wins = sum(delta > 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            ties = len(deltas) - wins - losses
            print(f"{mode:<52} {float(np.mean(deltas)) * 100:>8.1f}% [{low * 100:>6.1f}, {high * 100:>6.1f}] {wins}/{losses}/{ties:>5}")
        print("-" * 108)

    print("\nHIT@5 AND MRR@5 BY SOURCE")
    print("-" * 112)
    print(f"{'Retrieval mode':<52} {'Source':<42} {'N':>4} {'Hit@5':>9} {'MRR@5':>9} {'Recall@5':>10}")
    print("-" * 132)
    sources = sorted({case["source"] for case in cases})
    for mode, query_rankings in rankings.items():
        for source in sources:
            values = []
            for case, ordered_ids in zip(cases, query_rankings):
                if case["source"] != source:
                    continue
                values.append(_rank_metrics(ordered_ids, _gold_ids(case), 5))
            hits = [value[0] for value in values]
            reciprocal_ranks = [value[1] for value in values]
            recalls = [value[2] for value in values]
            hit_rate = float(np.mean(hits)) * 100 if hits else 0.0
            mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            print(f"{mode:<52} {source[:42]:<42} {len(values):>4} {hit_rate:>8.1f}% {mrr:>9.4f} {float(np.mean(recalls)) if recalls else 0.0:>10.4f}")
        print("-" * 132)
    print("\nHIT@5 BY QUERY TYPE")
    print(f"{'Retrieval mode':<52} {'Type':<18} {'N':>4} {'Hit@5':>9} {'MRR@5':>9}")
    print("-" * 100)
    query_types = sorted({str(case.get("kind", "gemini_or_legacy")) for case in cases})
    for mode, query_rankings in rankings.items():
        for query_type in query_types:
            values = [
                _rank_metrics(ordered_ids, _gold_ids(case), 5)
                for case, ordered_ids in zip(cases, query_rankings)
                if str(case.get("kind", "gemini_or_legacy")) == query_type
            ]
            hit_rate = float(np.mean([value[0] for value in values])) * 100 if values else 0.0
            mrr = float(np.mean([value[1] for value in values])) if values else 0.0
            print(f"{mode:<52} {query_type:<18} {len(values):>4} {hit_rate:>8.1f}% {mrr:>9.4f}")
        print("-" * 100)
    print("Synthetic cases use one known target; manual cases can label several relevant segments. Review misses and intervals before choosing a model.")
    print("=" * 132)


def _write_detailed_report(rankings, cases, fingerprint: str, corpus: list[dict[str, str]], retrieval_depth: int, latencies: dict[str, list[float]]) -> None:
    segment_map = {segment["id"]: segment["text"] for segment in corpus}
    output = {
        "corpus_fingerprint": fingerprint,
        "corpus_segments": len(corpus),
        "retrieval_depth": retrieval_depth,
        "query_count": len(cases),
        "production_pipeline": "BGE-M3 + BM25 weighted fusion (no reranker)",
        "reranker_latency_seconds_per_query": {
            mode: {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for mode, values in latencies.items() if values
        },
        "paired_hit_at_5_vs_production": {},
        "paired_hit_at_5_between_modes": {},
        "models": {},
    }
    baseline = PRODUCTION_MODE
    if baseline in rankings:
        base_hits = [
            _rank_metrics(ids, _gold_ids(case), 5)[0]
            for case, ids in zip(cases, rankings[baseline])
        ]
        clusters = [_case_cluster(case) for case in cases]
        for mode, query_rankings in rankings.items():
            if mode == baseline:
                continue
            candidate_hits = [
                _rank_metrics(ids, _gold_ids(case), 5)[0]
                for case, ids in zip(cases, query_rankings)
            ]
            deltas = [candidate - current for candidate, current in zip(candidate_hits, base_hits)]
            low, high = _bootstrap_interval(deltas, seed=RANDOM_SEED + 1, clusters=clusters)
            wins = sum(delta > 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            output["paired_hit_at_5_vs_production"][mode] = {
                "mean_delta": float(np.mean(deltas)),
                "ci_95": [low, high],
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
            }
    mode_names = list(rankings)
    clusters = [_case_cluster(case) for case in cases]
    for left_index, left_mode in enumerate(mode_names):
        left_hits = [
            _rank_metrics(ids, _gold_ids(case), 5)[0]
            for case, ids in zip(cases, rankings[left_mode])
        ]
        comparisons = {}
        for right_mode in mode_names[left_index + 1:]:
            right_hits = [
                _rank_metrics(ids, _gold_ids(case), 5)[0]
                for case, ids in zip(cases, rankings[right_mode])
            ]
            deltas = [left - right for left, right in zip(left_hits, right_hits)]
            low, high = _bootstrap_interval(deltas, seed=RANDOM_SEED + 2, clusters=clusters)
            wins = sum(delta > 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            comparisons[right_mode] = {
                "mean_delta": float(np.mean(deltas)),
                "ci_95": [low, high],
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
            }
        if comparisons:
            output["paired_hit_at_5_between_modes"][left_mode] = comparisons
    for mode, query_rankings in rankings.items():
        per_query = []
        for case, ordered_ids in zip(cases, query_rankings):
            gold = _gold_ids(case)
            hit, reciprocal_rank, recall = _rank_metrics(ordered_ids, gold, 5)
            ranks = {target: ordered_ids.index(target) + 1 if target in ordered_ids else None for target in sorted(gold)}
            per_query.append({
                "case_id": case.get("case_id"),
                "kind": case.get("kind", "gemini_or_legacy"),
                "query": case["query"],
                "source": case.get("source", ""),
                "relevant_ids": sorted(gold),
                "relevant_previews": {target: " ".join(segment_map.get(target, "").split())[:260] for target in sorted(gold)},
                "ranks": ranks,
                "hit_at_5": bool(hit),
                "reciprocal_rank_at_5": reciprocal_rank,
                "recall_at_5": recall,
                "top_10": [
                    {"id": segment_id, "preview": " ".join(segment_map.get(segment_id, "").split())[:260]}
                    for segment_id in ordered_ids[:10]
                ],
                "miss": not bool(hit),
            })
        output["models"][mode] = per_query
    REPORT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detailed per-query ranking report written to {REPORT_FILE}")


def run_test_corpus_benchmark(regenerate=False, skip_alternatives=False, manual_only=False, retrieval_depth: Optional[int] = None, skip_e5=False, skip_alt_reranker=False):
    import app_backend

    corpus, duplicates, fingerprint = _load_corpus()
    if not corpus:
        raise RuntimeError(f"No searchable documents found in {TEST_DIR}")
    sources = sorted({segment["source"] for segment in corpus})
    print(f"Loaded {len(corpus)} segments from {len(sources)} unique test documents")
    for duplicate in duplicates:
        print(f"Excluded duplicate: {duplicate}")

    generated_cases = _load_or_generate_dataset(corpus, fingerprint, regenerate, manual_only)
    manual_cases = _load_manual_cases(corpus, fingerprint)
    cases = generated_cases + manual_cases
    if not cases:
        raise RuntimeError("No benchmark queries found. Add cases to retrieval_manual_queries.json or generate Gemini queries.")

    passages = [search_utils.clean_markdown(segment["text"]) for segment in corpus]
    tokenized_corpus = [search_utils.tokenize(text) for text in passages]
    bm25 = BM25Okapi(tokenized_corpus)

    bge_vectors = np.asarray(app_backend.embedder.encode(
        passages,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
    ), dtype=np.float32)

    query_texts = [case["query"] for case in cases]
    e5_vectors = None
    e5_queries = None
    if not skip_alternatives and not skip_e5:
        print(f"Loading alternative embedding model: {ALTERNATIVE_EMBEDDING}")
        try:
            e5_model = SentenceTransformer(ALTERNATIVE_EMBEDDING)
            e5_vectors = _encode_documents(e5_model, passages, prefix="passage: ")
            e5_queries = np.asarray(e5_model.encode(
                [_query_input(query, "e5") for query in query_texts],
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            ), dtype=np.float32)
            del e5_model
        except Exception as error:
            print(f"Alternative embedding unavailable; continuing without it: {error}")
            e5_vectors = None
            e5_queries = None

    bge_queries = np.asarray(app_backend.embedder.encode(
        [_query_input(query, "bge") for query in query_texts],
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    ), dtype=np.float32)

    names = ["BM25 only", "BGE-M3 semantic", PRODUCTION_MODE, LEGACY_MINILM_MODE]
    print(f"Loading legacy reranker for comparison only: {LEGACY_RERANKER}")
    legacy_reranker = CrossEncoder(LEGACY_RERANKER)
    if e5_vectors is not None:
        names.extend(["E5-large semantic", "E5-large + BM25 (no reranker)", "E5-large + BM25 + MiniLM (comparison)"])
    alternative_reranker = None
    if not skip_alternatives and not skip_alt_reranker:
        print(f"Loading alternative reranker: {ALTERNATIVE_RERANKER}")
        try:
            alternative_reranker = CrossEncoder(ALTERNATIVE_RERANKER)
        except Exception as error:
            print(f"Alternative reranker unavailable; continuing without it: {error}")

    if alternative_reranker is not None:
        names.append("BGE-M3 + BM25 + BGE reranker v2-m3")
    rankings = {name: [] for name in names}
    latencies: dict[str, list[float]] = {name: [] for name in names if "reranker" in name.lower() or "minilm" in name.lower()}

    retrieval_depth = min(retrieval_depth or RETRIEVAL_DEPTH, len(corpus), 250)
    print(f"Hybrid fusion candidate depth: {retrieval_depth}")
    for query_index, case in enumerate(tqdm(cases, desc="Scoring model variants")):
        query = case["query"]
        bm25_scores = np.asarray(bm25.get_scores(search_utils.tokenize(query)), dtype=np.float32)
        bge_scores = bge_vectors @ bge_queries[query_index]
        bge_hybrid = _hybrid_candidates(query, bge_scores, bm25_scores, corpus, retrieval_depth)

        bm25_order = np.argsort(bm25_scores)[::-1]
        rankings["BM25 only"].append([
            corpus[index]["id"] for index in bm25_order if bm25_scores[index] > 0
        ][:max(TOP_K_VALUES)])
        rankings["BGE-M3 semantic"].append([
            corpus[index]["id"] for index in np.argsort(bge_scores)[::-1][:max(TOP_K_VALUES)]
        ])
        rankings[PRODUCTION_MODE].append([
            corpus[index]["id"] for index, _, _ in bge_hybrid
        ])
        start_time = time.perf_counter()
        rankings[LEGACY_MINILM_MODE].append(_rerank(query, bge_hybrid, corpus, legacy_reranker))
        latencies[LEGACY_MINILM_MODE].append(time.perf_counter() - start_time)
        if alternative_reranker is not None:
            alt_mode = "BGE-M3 + BM25 + BGE reranker v2-m3"
            start_time = time.perf_counter()
            rankings[alt_mode].append(_rerank(query, bge_hybrid, corpus, alternative_reranker))
            latencies[alt_mode].append(time.perf_counter() - start_time)

        if e5_vectors is not None and e5_queries is not None:
            e5_scores = e5_vectors @ e5_queries[query_index]
            e5_hybrid = _hybrid_candidates(query, e5_scores, bm25_scores, corpus, retrieval_depth)
            rankings["E5-large semantic"].append([
                corpus[index]["id"] for index in np.argsort(e5_scores)[::-1][:max(TOP_K_VALUES)]
            ])
            rankings["E5-large + BM25 (no reranker)"].append([
                corpus[index]["id"] for index, _, _ in e5_hybrid
            ])
            e5_mode = "E5-large + BM25 + MiniLM (comparison)"
            start_time = time.perf_counter()
            rankings[e5_mode].append(_rerank(query, e5_hybrid, corpus, legacy_reranker))
            latencies[e5_mode].append(time.perf_counter() - start_time)

    _evaluate_rankings(rankings, cases)
    _write_detailed_report(rankings, cases, fingerprint, corpus, retrieval_depth, latencies)
    return {"corpus_segments": len(corpus), "sources": sources, "query_cases": len(cases), "ranking_modes": names}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark search against the project's test documents.")
    parser.add_argument("--regenerate-queries", action="store_true", help="Generate new Gemini queries (requires GEMINI_API_KEY).")
    parser.add_argument("--skip-alternatives", action="store_true", help="Skip E5 and the BGE multilingual reranker comparison.")
    parser.add_argument("--skip-e5", action="store_true", help="Skip only the multilingual E5 embedding comparison.")
    parser.add_argument("--skip-bge-reranker", action="store_true", help="Skip only the multilingual BGE reranker comparison.")
    parser.add_argument("--manual-only", action="store_true", help="Use only the manually labeled query set.")
    parser.add_argument("--retrieval-depth", type=int, default=RETRIEVAL_DEPTH, help="Candidates for hybrid fusion (default 40, matching app search for top-5).")
    parser.add_argument("--export-corpus", type=Path, help="Export stable segment IDs and text to JSON for writing manual labels.")
    args = parser.parse_args()

    if args.export_corpus:
        export_corpus, excluded, export_fingerprint = _load_corpus()
        args.export_corpus.parent.mkdir(parents=True, exist_ok=True)
        args.export_corpus.write_text(json.dumps({
            "fingerprint": export_fingerprint,
            "segments": export_corpus,
            "excluded_duplicates": excluded,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Exported {len(export_corpus)} segments to {args.export_corpus}")
    else:
        run_test_corpus_benchmark(
            regenerate=args.regenerate_queries,
            skip_alternatives=args.skip_alternatives,
            manual_only=args.manual_only,
            retrieval_depth=args.retrieval_depth,
            skip_e5=args.skip_e5,
            skip_alt_reranker=args.skip_bge_reranker,
        )
