"""Run the RAG system on the gold set and compute evaluation metrics."""

import json
from pathlib import Path

from rag import run
from retrieval import Retriever, load_config


ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "tests" / "gold-set.jsonl"
RESULTS_PATH = ROOT / "reports" / "lab-5" / "results.jsonl"
SUMMARY_PATH = ROOT / "reports" / "lab-5" / "summary.json"


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_missing(gold):
    done = {r["id"] for r in load_jsonl(RESULTS_PATH)}
    todo = [item for item in gold if item["id"] not in done]
    if not todo:
        return
    config = load_config()["generation"]
    retriever = Retriever()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(todo, start=1):
        print(f"[{n}/{len(todo)}] {item['id']}: {item['query']}")
        response = run(item["query"], retriever, config)
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["id"], **response}, ensure_ascii=False) + "\n")


def share(values):
    return round(sum(values) / len(values), 2) if values else None


def evaluate(gold, results):
    by_id = {r["id"]: r for r in results}
    rows = []
    for item in gold:
        r = by_id[item["id"]]
        found = {s["chunk_id"] for s in r["sources"]}
        cited = {c["chunk_id"] for c in r["citations"]}
        relevant = set(item["relevant_chunk_ids"])
        rows.append({
            "id": item["id"], "type": item["type"], "status": r["status"],
            "attempts": r["attempts"], "time_s": r["retrieval_s"] + r["generation_s"],
            "recall5": bool(relevant & found) if relevant else None,
            "cited_relevant": bool(relevant & cited) if relevant else None,
        })

    summary = {}
    for group in ("simple", "complex", "all"):
        positive = [r for r in rows if r["type"] != "negative" and group in ("all", r["type"])]
        answered = [r for r in positive if r["status"] != "refusal"]
        summary[group] = {
            "queries": len(positive),
            "recall@5": share([r["recall5"] for r in positive]),
            "valid_citations": share([r["status"] == "ok" for r in answered]),
            "false_refusals": share([r["status"] == "refusal" for r in positive]),
            "cited_relevant": share([r["cited_relevant"] for r in answered]),
        }
    negative = [r for r in rows if r["type"] == "negative"]
    summary["negative"] = {"queries": len(negative),
                           "correct_refusals": share([r["status"] == "refusal" for r in negative])}
    summary["avg_time_s"] = round(sum(r["time_s"] for r in rows) / len(rows), 1)
    return rows, summary


def main():
    gold = load_jsonl(GOLD_PATH)
    run_missing(gold)
    rows, summary = evaluate(gold, load_jsonl(RESULTS_PATH))

    print(f"\n{'id':<5}{'type':<10}{'recall5':>8}{'status':>12}{'tries':>7}{'cited_rel':>11}")
    for r in rows:
        mark = lambda v: "-" if v is None else ("yes" if v else "no")
        print(f"{r['id']:<5}{r['type']:<10}{mark(r['recall5']):>8}{r['status']:>12}"
              f"{r['attempts']:>7}{mark(r['cited_relevant']):>11}")

    print("\n" + json.dumps(summary, indent=2))
    SUMMARY_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()