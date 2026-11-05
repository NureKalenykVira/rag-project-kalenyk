"""Compare Recall@5 and Recall@10 of dense, BM25 and hybrid search on the eval set."""

import json
from pathlib import Path

from retrieval import Retriever


ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "tests" / "eval-set.jsonl"
OUTPUT_PATH = ROOT / "reports" / "lab-4" / "retrieval-eval.json"
MAX_K = 10


def first_relevant_rank(hits, relevant_ids):
    for hit in hits:
        if hit["chunk_id"] in relevant_ids:
            return hit["rank"]
    return None


def recall_at(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f]

    retriever = Retriever()
    methods = {"dense": retriever.dense_search, "bm25": retriever.bm25_search,
               "hybrid": retriever.hybrid_search, "rerank": retriever.rerank_search}
    
    ranks = {name: [] for name in methods}
    for item in eval_set:
        for name, search in methods.items():
            hits = search(item["query"], MAX_K)
            ranks[name].append(first_relevant_rank(hits, item["relevant_chunk_ids"]))

    print(f"{'id':<5}{'dense':>7}{'bm25':>7}{'hybrid':>8}{'rerank':>8}  query")
    for i, item in enumerate(eval_set):
        row = [ranks[name][i] or "-" for name in methods]
        print(f"{item['id']:<5}{row[0]!s:>7}{row[1]!s:>7}{row[2]!s:>8}{row[3]!s:>8}  {item['query']}")
        
    summary = {}
    print()
    for name in methods:
        summary[name] = {"recall@5": recall_at(ranks[name], 5),
                         "recall@10": recall_at(ranks[name], 10)}
        print(f"{name:<7} Recall@5: {summary[name]['recall@5']:.2f}  "
              f"Recall@10: {summary[name]['recall@10']:.2f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "ranks": ranks,
              "queries": [item["id"] for item in eval_set]}
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()