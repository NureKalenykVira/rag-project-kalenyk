"""Measure Recall@5 and Recall@10 on the evaluation set."""

import argparse
import json
from pathlib import Path

from search import load_config, load_search, search


ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "tests" / "eval-set.jsonl"
RESULTS_DIR = ROOT / "reports" / "lab-3"
MAX_K = 10


def load_eval_set():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def first_relevant_rank(found_ids, relevant_ids):
    for rank, chunk_id in enumerate(found_ids, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def recall_at(results, k):
    hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
    return hits / len(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the eval set")
    parser.add_argument("--name", default="baseline", help="experiment name")
    args = parser.parse_args()

    model, collection = load_search()
    eval_set = load_eval_set()

    results = []
    for item in eval_set:
        hits = search(model, collection, item["query"], MAX_K)
        found_ids = [h["chunk_id"] for h in hits]
        rank = first_relevant_rank(found_ids, item["relevant_chunk_ids"])
        results.append({"id": item["id"], "query": item["query"],
                        "rank": rank, "found_ids": found_ids})

    print(f"Experiment: {args.name}\n")
    for r in results:
        rank = r["rank"] if r["rank"] is not None else "-"
        print(f"{r['id']}  rank={rank!s:>2}  {r['query']}")

    recall5, recall10 = recall_at(results, 5), recall_at(results, 10)
    print(f"\nRecall@5:  {recall5:.2f}")
    print(f"Recall@10: {recall10:.2f}")

    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"name": args.name, "recall@5": recall5, "recall@10": recall10,
              "embedding": config["embedding"], "index": config["index"],
              "results": results}
    output = RESULTS_DIR / f"eval-{args.name}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()