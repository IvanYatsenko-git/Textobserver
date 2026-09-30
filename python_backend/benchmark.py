import os
import json
import random
import re
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

try:
    from google import genai
except ImportError:
    genai = None

# Імпортуємо ваш готовий бекенд
import app_backend
from search_utils import STOPWORDS

# ─── Налаштування ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATASET_FILE = "benchmark_dataset.json"
NUM_SAMPLES = 20
TOP_K_VALUES = (5, 10, 15, 20, 30)
SEARCH_MODES = {
    "semantic_only": "Semantic only (BGE-M3)",
    "bm25_only": "Lexical only (BM25)",
    "hybrid_no_reranker": "Production: BGE-M3 + BM25 (weighted fusion)",
}

# Пряма ініціалізація клієнта
client = None
if genai is not None and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Помилка створення клієнта: {e}")


def extract_keywords(text: str, top_n: int = 8) -> str:
    """Будує локальний пошуковий запит із ключових слів тексту без зовнішнього API."""
    cleaned = re.sub(r"[^\w\s\-]", " ", text.lower(), flags=re.UNICODE)
    tokens = [token for token in cleaned.split() if len(token) > 2 and token not in STOPWORDS]
    if not tokens:
        return ""

    counts = Counter(tokens)
    keywords = [word for word, _ in counts.most_common(top_n)]
    return " ".join(keywords)


def generate_local_dataset():
    """Генерує простий benchmark без Gemini: запити складаються з ключових слів сегмента."""
    print("Отримання сегментів з бази даних...")
    data = app_backend.collection.get(include=["documents"])
    ids = data.get('ids', [])
    docs = data.get('documents', [])

    if not ids:
        print("❌ База порожня! Спочатку відкрийте та проіндексуйте файл у додатку.")
        return []

    valid_pairs = [(i, d) for i, d in zip(ids, docs) if len(str(d).strip()) >= 50]
    if not valid_pairs:
        print("❌ Немає валідних довгих сегментів для тестування.")
        return []

    samples = random.sample(valid_pairs, min(NUM_SAMPLES, len(valid_pairs)))
    dataset = []

    print(f"Генерація {len(samples)} локальних запитів із ключових слів...")
    for doc_id, text in tqdm(samples):
        query = extract_keywords(text.strip(), top_n=8)
        if not query:
            continue
        dataset.append({
            "query": query,
            "expected_id": doc_id,
            "text_preview": text[:80].strip() + "..."
        })

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return dataset


def generate_synthetic_dataset():
    """Генерує пошукові запити до існуючих сегментів через новий SDK."""
    if not client:
        print("Gemini API key не заданий у змінній середовища GEMINI_API_KEY; використовую локальний датасет.")
        return []

    print("Отримання сегментів з бази даних...")
    data = app_backend.collection.get(include=["documents"])
    ids = data.get('ids', [])
    docs = data.get('documents', [])

    if not ids:
        print("❌ База порожня! Спочатку відкрийте та проіндексуйте файл у додатку.")
        return []

    valid_pairs = [(i, d) for i, d in zip(ids, docs) if len(d.strip()) >= 50]
    if not valid_pairs:
        print("❌ Немає валідних довгих сегментів для тестування.")
        return []

    samples = random.sample(valid_pairs, min(NUM_SAMPLES, len(valid_pairs)))
    dataset = []

    print(f"Генерація {len(samples)} синтетичних запитів через Gemini...")
    for doc_id, text in tqdm(samples):
        prompt = (
            "Уяви, що ти студент або дослідник, який шукає цей текст. "
            "Напиши ОДИН реалістичний пошуковий запит (3-8 слів) українською мовою, "
            "за яким людина хотіла б знайти цей абзац. Не пиши нічого зайвого, тільки сам запит.\n\n"
            f"Текст:\n{text.strip()}\n\nЗапит:"
        )
        
        try:
            # Використовуємо актуальну модель gemini-2.5-flash
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            query = response.text.strip().replace('"', '').replace('\n', ' ')
            dataset.append({
                "query": query,
                "expected_id": doc_id,
                "text_preview": text[:80].strip() + "..."
            })
        except Exception as e:
            print(f"\nПомилка генерації для {doc_id}: {e}")

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    return dataset


def dataset_matches_current_index(dataset):
    """Reject benchmark labels whose segment IDs or preview text are stale."""
    if not dataset:
        return False

    indexed = app_backend.collection.get(include=["documents"])
    documents_by_id = dict(zip(indexed.get("ids", []), indexed.get("documents", [])))

    for item in dataset:
        expected_id = str(item.get("expected_id", ""))
        current_text = documents_by_id.get(expected_id)
        if current_text is None:
            return False

        preview = str(item.get("text_preview", "")).rstrip("….").strip()
        normalized_text = re.sub(r"\s+", " ", str(current_text)).strip().lower()
        normalized_preview = re.sub(r"\s+", " ", preview).strip().lower()
        if normalized_preview and not normalized_text.startswith(normalized_preview):
            return False

    return True

def run_comparison(dataset):
    """Compare retrieval stages on identical queries and relevance labels."""
    if not dataset:
        print("Немає запитів для оцінювання.")
        return

    top_ks = tuple(value for value in TOP_K_VALUES if value > 0)
    max_k = max(top_ks)
    metrics = {
        mode: {top_k: {"hits": 0, "reciprocal_ranks": []} for top_k in top_ks}
        for mode in SEARCH_MODES
    }

    for mode, label in SEARCH_MODES.items():
        print(f"\nТест: {label} ({len(dataset)} запитів, Top-{max_k})")
        for item in tqdm(dataset):
            query = str(item.get("query", "")).strip()
            expected_id = str(item.get("expected_id", ""))
            if not query or not expected_id:
                continue

            results = app_backend.benchmark_search(query, max_k, mode)
            result_ids = [str(result["id"]) for result in results]
            rank = result_ids.index(expected_id) + 1 if expected_id in result_ids else 0

            for top_k in top_ks:
                found_rank = rank if rank <= top_k else 0
                current = metrics[mode][top_k]
                current["hits"] += int(found_rank > 0)
                current["reciprocal_ranks"].append(1.0 / found_rank if found_rank else 0.0)

    sample_count = len(dataset)
    print("\n" + "=" * 96)
    print(f"ПОРІВНЯННЯ ПОШУКОВИХ РЕЖИМІВ | {sample_count} однакових запитів")
    print("=" * 96)
    print(f"{'Режим пошуку':<31} {'K':>3} {'Hit Rate':>12} {'MRR':>10}")
    print("-" * 96)

    for mode, label in SEARCH_MODES.items():
        for top_k in top_ks:
            current = metrics[mode][top_k]
            hit_rate = current["hits"] / sample_count * 100
            mrr = float(np.mean(current["reciprocal_ranks"])) if current["reciprocal_ranks"] else 0.0
            print(f"{label:<31} {top_k:>3} {hit_rate:>11.1f}% {mrr:>10.4f}")
        print("-" * 96)

    print("Hit Rate показує, чи знайдено очікуваний сегмент у Top-K; MRR враховує його позицію.")
    print("=" * 96)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare retrieval quality on the active index or the project test corpus.")
    parser.add_argument(
        "--test-corpus",
        action="store_true",
        help="Benchmark the independent Markdown/DOCX/PDF documents in ../test",
    )
    parser.add_argument(
        "--regenerate-test-queries",
        action="store_true",
        help="Regenerate test-corpus queries through Gemini instead of reusing the saved set",
    )
    parser.add_argument(
        "--skip-alternative-models",
        action="store_true",
        help="Skip multilingual-E5 and BGE reranker downloads/inference",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Use only manually labeled cases from test/retrieval_manual_queries.json",
    )
    args = parser.parse_args()

    if args.test_corpus:
        from test_corpus_benchmark import run_test_corpus_benchmark

        run_test_corpus_benchmark(
            regenerate=args.regenerate_test_queries,
            skip_alternatives=args.skip_alternative_models,
            manual_only=args.manual_only,
        )
        raise SystemExit(0)

    dataset = []

    if not os.path.exists(DATASET_FILE):
        dataset = generate_synthetic_dataset()
        if not dataset:
            dataset = generate_local_dataset()
    else:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not dataset_matches_current_index(dataset):
            print("Датасет не відповідає поточному індексу; генерую його заново.")
            dataset = generate_synthetic_dataset()
            if not dataset:
                dataset = generate_local_dataset()
        elif not dataset:
            dataset = generate_local_dataset()
        else:
            print(f"Завантажено готовий датасет із {len(dataset)} запитів.")

    if dataset:
        run_comparison(dataset)
    else:
        print("❌ Не вдалося зібрати жодного тестового запиту для benchmark.")
