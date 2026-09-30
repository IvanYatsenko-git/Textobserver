import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Set

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app_backend


DEFAULT_THESES = {
    "Теза 1": "Проаналізувати предметну область, існуючі методи виявлення дезінформації та формалізувати функціональні й нефункціональні вимоги до інформаційної системи.",
    "Теза 2": "Дослідити підходи до виявлення дезінформації в соціальних мережах та оцінити їх ефективність.",
    "Теза 3": "Описати архітектуру інформаційної системи для перевірки фактів на основі зовнішніх джерел.",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gold_map(raw_gold) -> Dict[str, Set[str]]:
    if raw_gold is None:
        return {}

    result: Dict[str, Set[str]] = {}
    if isinstance(raw_gold, dict):
        for thesis, values in raw_gold.items():
            if isinstance(values, list):
                result[thesis] = {str(v) for v in values}
            elif values is None:
                result[thesis] = set()
    return result


def evaluate_query(thesis_text: str, gold_ids: Set[str], top_k: int = 10, thresholds: Iterable[int] = (0, 10, 20, 30, 40, 50, 60, 70)) -> List[dict]:
    response = app_backend.cmd_search({"query": thesis_text, "n_results": top_k * 4})
    if response.get("status") != "success":
        return []

    results = response.get("results", [])
    if not results:
        return []

    rows = []
    gold_set = set(gold_ids)
    for thresh in thresholds:
        predicted = [item for item in results if float(item.get("similarity", 0)) >= float(thresh)]
        predicted = predicted[:top_k]
        pred_ids = {str(item["id"]) for item in predicted}
        tp = len(pred_ids & gold_set)
        precision = (tp / len(pred_ids)) if pred_ids else 0.0
        recall = (tp / len(gold_set)) if gold_set else 0.0
        coverage = (tp / len(gold_set)) if gold_set else 0.0
        rows.append({
            "threshold": thresh,
            "top_k": top_k,
            "predicted_count": len(predicted),
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "coverage": round(coverage, 4),
            "predicted_ids": sorted(pred_ids),
        })
    return rows


def print_table(rows: List[dict]):
    headers = ["threshold", "predicted", "TP", "precision", "recall", "coverage"]
    print("\n" + " | ".join(headers))
    print("-" * 90)
    for row in rows:
        print(f"{row['threshold']:>9} | {row['predicted_count']:>9} | {row['tp']:>2} | {row['precision']:.4f} | {row['recall']:.4f} | {row['coverage']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate thesis-based search quality using precision@k and recall.")
    parser.add_argument("--theses-file", help="JSON file with thesis texts or thesis->list mapping")
    parser.add_argument("--gold-file", help="JSON file with gold relevant segment IDs per thesis")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--thresholds", nargs="*", type=int, default=[0, 10, 20, 30, 40, 50, 60, 70])
    args = parser.parse_args()

    if args.theses_file:
        raw = load_json(args.theses_file)
        if isinstance(raw, list):
            thesis_map = {f"thesis_{i + 1}": str(item) for i, item in enumerate(raw)}
        elif isinstance(raw, dict):
            thesis_map = {str(k): str(v) for k, v in raw.items()}
        else:
            thesis_map = DEFAULT_THESES
    else:
        thesis_map = DEFAULT_THESES

    if args.gold_file:
        gold_map = normalize_gold_map(load_json(args.gold_file))
    else:
        gold_map = {}

    if not thesis_map:
        print("No theses provided.")
        return

    print("Evaluating thesis retrieval quality...\n")
    for thesis_name, thesis_text in thesis_map.items():
        gold_ids = set(gold_map.get(thesis_name, gold_map.get(thesis_text, [])))
        if not gold_ids:
            print(f"[{thesis_name}] no gold labels provided; skipping.")
            continue

        print(f"=== {thesis_name} ===")
        rows = evaluate_query(thesis_text, gold_ids, top_k=args.top_k, thresholds=args.thresholds)
        if not rows:
            print("No results returned from backend.")
            continue
        print_table(rows)
        print()

    print("Tip: provide a JSON gold file like {\"Thesis 1\": [\"seg-1\", \"seg-2\"]} to measure real precision/recall.")


if __name__ == "__main__":
    main()
