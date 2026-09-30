import sys
import json
import os
import re
import chromadb
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from search_utils import tokenize as tokenize_text, keyword_overlap_score

# Вимикаємо зайві виводи в stdout
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Абсолютний шлях до БД
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("TEXT_OBSERVER_DATA_DIR", BASE_DIR)
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db_storage")

print("[Python] Ініціалізація моделей (це може зайняти хвилину під час першого запуску)...", file=sys.stderr)

# 1. Ембединг модель (семантичний пошук)
embedder = SentenceTransformer('BAAI/bge-m3')

# 2. Підключення до ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="document_segments", 
    metadata={"hnsw:space": "cosine"} 
)
print('__TEXTOBSERVER_READY__', file=sys.stderr, flush=True)

# ─── Глобальний стан ────────────────────────────────────────────────────────────

GLOBAL_BM25: Optional[BM25Okapi] = None
GLOBAL_BM25_IDS: List[str] = []
BM25_NEEDS_REBUILD: bool = True
DB_VERSION: int = 0  # Для скидання кешу при зміні бази

# ─── Типи даних (Dataclasses) ───────────────────────────────────────────────────

@dataclass
class Candidate:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0

# ─── Утиліти ────────────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    """Покращене очищення тексту від розмітки Markdown, HTML та іншого сміття."""
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'\|.*?\|', ' ', text)
    text = re.sub(r'[-*]{3,}', ' ', text)
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'[*_~`>]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> List[str]:
    """Якісна токенізація з підтримкою Unicode для BM25."""
    return tokenize_text(text)

def build_bm25_from_chroma() -> None:
    """Відновлює BM25 індекс, але не дублює текст у пам'яті."""
    global GLOBAL_BM25, GLOBAL_BM25_IDS, BM25_NEEDS_REBUILD
    data = collection.get(include=["documents"])
    ids = data.get('ids', [])
    docs = data.get('documents', [])
    
    if not ids:
        return
        
    tokenized_corpus = [tokenize(doc) for doc in docs]
    GLOBAL_BM25 = BM25Okapi(tokenized_corpus)
    GLOBAL_BM25_IDS = ids
    BM25_NEEDS_REBUILD = False

# ─── Логіка пошуку (Розділена на окремі функції) ───────────────────────────────

def get_semantic_scores(query: str, top_k: int) -> Dict[str, float]:
    """Виконує векторний пошук через BGE-M3."""
    # BGE-M3 is trained for direct query/document matching; unlike E5 it does
    # not require a query instruction prefix.
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
    
    chroma_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["distances"]
    )
    
    if not chroma_results or not chroma_results['ids']:
        return {}
        
    return {
        doc_id: 1.0 - distance 
        for doc_id, distance in zip(chroma_results['ids'][0], chroma_results['distances'][0])
    }

def get_lexical_scores(query: str, top_k: int) -> Dict[str, float]:
    """Виконує лексичний пошук через BM25."""
    if GLOBAL_BM25 is None:
        return {}
        
    tokenized_query = tokenize(query)
    raw_scores = GLOBAL_BM25.get_scores(tokenized_query)
    
    top_indices = np.argsort(raw_scores)[::-1][:top_k]
    return {
        GLOBAL_BM25_IDS[idx]: raw_scores[idx] 
        for idx in top_indices if raw_scores[idx] > 0
    }

def weighted_hybrid_search(semantic_scores: Dict[str, float], bm25_scores: Dict[str, float], top_k: int, query: str) -> List[Candidate]:
    """Зважене злиття semantic + BM25 з більш лояльним фільтром для довгих тез."""
    all_ids = sorted(set(semantic_scores.keys()).union(set(bm25_scores.keys())))
    if not all_ids:
        return []

    cand_data = collection.get(ids=all_ids, include=["documents", "metadatas"])
    docs_dict = dict(zip(cand_data['ids'], cand_data['documents']))
    meta_dict = dict(zip(cand_data['ids'], cand_data['metadatas']))

    semantic_max = max(semantic_scores.values(), default=0.0)
    bm25_max = max(bm25_scores.values(), default=0.0)

    query_terms = set(tokenize(query))
    candidates = []
    for doc_id in all_ids:
        text = docs_dict.get(doc_id, "")
        semantic_score = semantic_scores.get(doc_id, 0.0)
        bm25_score = bm25_scores.get(doc_id, 0.0)
        overlap = keyword_overlap_score(query, text)

        # Для довгих тез не відсікаємо сегменти лише через відсутність буквального overlap.
        # Якщо документ має сильний semantic/BM25 сигнал, він все одно має шанс потрапити в топ.
        if query_terms and overlap <= 0.0 and semantic_score <= 0.0 and bm25_score <= 0.0:
            continue

        semantic_norm = semantic_score / semantic_max if semantic_max > 0 else 0.0
        bm25_norm = bm25_score / bm25_max if bm25_max > 0 else 0.0

        # Калібрування для технічних/дослідницьких текстів: збільшуємо вагу лексичного збігу,
        # не відкидаючи семантику, але робимо пошук менш “надто семантичним”.
        hybrid_score = (0.35 * semantic_norm) + (0.35 * bm25_norm) + (0.30 * overlap)

        candidates.append(Candidate(
            id=doc_id,
            text=text,
            metadata=meta_dict.get(doc_id, {}),
            score=hybrid_score
        ))

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:top_k]

def candidates_to_results(candidates: List[Candidate], n_results: int) -> List[Dict[str, Any]]:
    """Return weighted-fusion candidates in the response format used by the app."""
    return [
        {
            "id": candidate.id,
            "text": candidate.text,
            "similarity": round(candidate.score * 100, 1),
            "metadata": candidate.metadata,
        }
        for candidate in candidates[:n_results]
    ]

@lru_cache(maxsize=128)
def cached_search(query: str, n_results: int, top_k_retrieval: int, db_version: int) -> List[Dict[str, Any]]:
    """Обгортка для LRU-кешування гібридного пошуку."""
    semantic_scores = get_semantic_scores(query, top_k_retrieval)
    lexical_scores = get_lexical_scores(query, top_k_retrieval)

    # Якщо запит дуже загальний, не даємо семантичному пошуку затьмарити все.
    if not query.strip():
        return []

    hybrid_candidates = weighted_hybrid_search(semantic_scores, lexical_scores, top_k_retrieval, query)
    if hybrid_candidates:
        return candidates_to_results(hybrid_candidates, n_results)

    # Fallback: якщо гібридний пошук нічого не дав, повертаємо топові документи за лексичним перекриттям.
    all_data = collection.get(include=["documents", "metadatas"])
    fallback_candidates: List[Candidate] = []
    for doc_id, text, meta in zip(all_data.get('ids', []), all_data.get('documents', []), all_data.get('metadatas', [])):
        if not text:
            continue
        overlap = keyword_overlap_score(query, text)
        if overlap > 0:
            fallback_candidates.append(Candidate(id=str(doc_id), text=str(text), metadata=meta or {}, score=float(overlap)))

    if fallback_candidates:
        fallback_candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates_to_results(fallback_candidates, n_results)

    return []

# ─── Команди ────────────────────────────────────────────────────────────────────

def cmd_clear(data: Dict[str, Any]) -> Dict[str, Any]:
    global GLOBAL_BM25, GLOBAL_BM25_IDS, collection, BM25_NEEDS_REBUILD, DB_VERSION
    try:
        chroma_client.delete_collection(name="document_segments")
        collection = chroma_client.get_or_create_collection(
            name="document_segments",
            metadata={"hnsw:space": "cosine"}
        )
        GLOBAL_BM25 = None
        GLOBAL_BM25_IDS = []
        BM25_NEEDS_REBUILD = True
        
        # Скидаємо кеш
        DB_VERSION += 1
        cached_search.cache_clear()
        
        print("[Python] Колекцію очищено", file=sys.stderr)
        return {"status": "success", "message": "Колекцію очищено"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def cmd_index(data: Dict[str, Any]) -> Dict[str, Any]:
    global BM25_NEEDS_REBUILD, DB_VERSION
    chunks = data.get("chunks", [])
    source_file = data.get("source_file", "unknown.md")
    
    if not chunks:
        return {"status": "error", "message": "No chunks provided"}
    
    ids, original_texts, clean_texts, metadatas = [], [], [], []

    # Оптимізація: уникаємо створення зайвих масивів (менше копіювання рядків)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
            
        raw_text = chunk.get("text")
        if not raw_text or not str(raw_text).strip():
            continue
            
        text = str(raw_text).strip()
        ids.append(str(chunk.get("id")))
        original_texts.append(text)
        clean_texts.append(clean_markdown(text))
        
        meta = {"source": source_file}
        # Записуємо лише ті метадані, які існують у JSON-запиті
        for key in ("startChar", "endChar", "heading", "sentenceCount"):
            val = chunk.get(key)
            if val is not None:
                meta[key] = val
        metadatas.append(meta)
            
    if not ids:
        return {"status": "success", "message": "Немає валідних чанків", "indexed": 0}

    print(f"[Python] Кодування {len(ids)} сегментів батчами...", file=sys.stderr)
    vectors = embedder.encode(clean_texts, normalize_embeddings=True, batch_size=32).tolist()

    try:
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=original_texts,
            metadatas=metadatas
        )
        BM25_NEEDS_REBUILD = True 
        
        # Скидаємо кеш після зміни бази
        DB_VERSION += 1
        cached_search.cache_clear()
        
        return {
            "status": "success", 
            "message": f"Проіндексовано {len(ids)} сегментів", 
            "indexed": len(ids)
        }
    except Exception as e:
        print(f"[Python Warning] Помилка індексації: {str(e)}", file=sys.stderr)
        return {"status": "error", "message": str(e)}

def cmd_search(data: Dict[str, Any]) -> Dict[str, Any]:
    global GLOBAL_BM25, BM25_NEEDS_REBUILD, DB_VERSION
    query = str(data.get("query", "")).strip()
    n_results = int(data.get("n_results", 5))

    if not query:
        return {"status": "error", "message": "Порожній запит"}

    count = collection.count()
    if count == 0:
        return {"status": "error", "message": "База даних порожня. Відкрийте файл для індексації."}

    if GLOBAL_BM25 is None or BM25_NEEDS_REBUILD:
        build_bm25_from_chroma()

    # Якщо JS просить оцінити весь документ (n_results >= count)
    if n_results >= count:
        top_k_retrieval = count # Беремо абсолютно всі сегменти
    else:
        # Трохи ширший candidate pool для технічних тез: більше шансів, що релевантний фрагмент
        # потрапить у злитий список, навіть якщо на верхніх позиціях є близькі, але не ідеальні варіанти.
        top_k_retrieval = min(max(n_results * 8, 40), count, 250)
    
    # Викликаємо пошук із LRU-кешуванням
    final_results = cached_search(query, n_results, top_k_retrieval, DB_VERSION)

    print(f"[Python] Запит '{query[:40]}' → відправлено {len(final_results)} результатів", file=sys.stderr)
    return {"status": "success", "results": final_results}


def benchmark_search(query: str, n_results: int, mode: str) -> List[Dict[str, Any]]:
    """Run one production retrieval stage in isolation for comparative benchmarks."""
    query = str(query).strip()
    if not query or collection.count() == 0:
        return []

    if GLOBAL_BM25 is None or BM25_NEEDS_REBUILD:
        build_bm25_from_chroma()

    count = collection.count()
    retrieval_depth = min(max(n_results * 8, 40), count, 250)

    if mode == "full_pipeline":
        result = cmd_search({"query": query, "n_results": n_results})
        return result.get("results", []) if result.get("status") == "success" else []

    if mode == "semantic_only":
        ranked_ids = sorted(
            get_semantic_scores(query, retrieval_depth).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        scores = dict(ranked_ids)
    elif mode == "bm25_only":
        ranked_ids = sorted(
            get_lexical_scores(query, retrieval_depth).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        scores = dict(ranked_ids)
    elif mode == "hybrid_no_reranker":
        semantic_scores = get_semantic_scores(query, retrieval_depth)
        lexical_scores = get_lexical_scores(query, retrieval_depth)
        candidates = weighted_hybrid_search(semantic_scores, lexical_scores, retrieval_depth, query)
        return [
            {
                "id": candidate.id,
                "text": candidate.text,
                "similarity": round(candidate.score * 100, 1),
                "metadata": candidate.metadata,
            }
            for candidate in candidates[:n_results]
        ]
    else:
        raise ValueError(f"Unknown benchmark search mode: {mode}")

    ids = [doc_id for doc_id, _ in ranked_ids[:n_results]]
    if not ids:
        return []

    documents = collection.get(ids=ids, include=["documents", "metadatas"])
    docs_by_id = dict(zip(documents["ids"], documents["documents"]))
    metadata_by_id = dict(zip(documents["ids"], documents["metadatas"]))
    return [
        {
            "id": doc_id,
            "text": docs_by_id.get(doc_id, ""),
            "similarity": float(scores.get(doc_id, 0.0)),
            "metadata": metadata_by_id.get(doc_id, {}),
        }
        for doc_id in ids
    ]

# ─── Головний цикл stdin/stdout ───────────────────────────────────────────────

COMMANDS = {
    "index": cmd_index,
    "search": cmd_search,
    "clear": cmd_clear,
}


def run_cli_server() -> None:
    """Запускає JSON stdin/stdout CLI, коли файл запускається напряму."""
    print("[Python] Бекенд готовий. Чекаємо на команди...", file=sys.stderr)
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            command = request.get("command")
            data = request.get("data", {})

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            handler = COMMANDS.get(command)
            if handler:
                response = handler(data)
            else:
                response = {"status": "error", "message": f"Невідома команда: {command}"}

        except json.JSONDecodeError as e:
            response = {"status": "error", "message": f"JSON помилка: {e}"}
        except Exception as e:
            response = {"status": "error", "message": str(e)}

        print(json.dumps(response, ensure_ascii=False))
        sys.stdout.flush()


if __name__ == "__main__":
    run_cli_server()
