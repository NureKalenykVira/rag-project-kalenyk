"""RAG system CLI: retrieve, generate with citations, validate, print or save as JSON."""

import argparse
import json
import time
from pathlib import Path

from contract import BLOCKED, COMMUNITY_ADVICE, is_english, looks_like_injection
from generate import answer_question
from retrieval import Retriever, load_config


def blocked_response(question, config, sources=None, retrieval_s=0.0, generation_s=0.0):
    return {
        "question": question,
        "answer": BLOCKED,
        "status": "blocked",
        "attempts": 0 if sources is None else 1,
        "advice": None,
        "citations": [],
        "sources": sources or [],
        "model": config["model"],
        "retrieval_s": round(retrieval_s, 1),
        "generation_s": round(generation_s, 1),
    }


def run(question, retriever, config):
    if looks_like_injection(question):
        return blocked_response(question, config)

    start = time.perf_counter()
    hits = retriever.rerank_search(question, config["top_k"])
    retrieval_s = time.perf_counter() - start

    start = time.perf_counter()
    result = answer_question(question, hits, config)
    generation_s = time.perf_counter() - start

    sources = [{"number": n, "chunk_id": h["chunk_id"], "doc_id": h["doc_id"],
                "title": h["title"], "score": round(h["score"], 3)}
               for n, h in enumerate(hits, start=1)]

    if not is_english(result["answer"]):
        return blocked_response(question, config, sources, retrieval_s, generation_s)

    cited = [s for s in sources if s["chunk_id"] in result["cited_chunks"]]
    return {
        "question": question,
        "answer": result["answer"],
        "status": result["status"],
        "attempts": result["attempts"],
        "advice": COMMUNITY_ADVICE if result["status"] == "refusal" else None,
        "citations": cited,
        "sources": sources,
        "model": config["model"],
        "retrieval_s": round(retrieval_s, 1),
        "generation_s": round(generation_s, 1),
    }


def print_response(response):
    print(f"\nQuestion: {response['question']}\n")
    if response["status"] == "uncertain":
        print("WARNING: the answer could not be verified against the sources.\n")
    print(response["answer"] + "\n")
    if response["advice"]:
        print(response["advice"] + "\n")

    if response["citations"]:
        print("Cited sources:")
        for source in response["citations"]:
            print(f"  [{source['number']}] {source['title']} - {source['chunk_id']}")
    print(f"\nStatus: {response['status']}, attempts: {response['attempts']}, "
          f"model: {response['model']}, retrieval: {response['retrieval_s']} s, "
          f"generation: {response['generation_s']} s")


def interactive(retriever, config):
    print("Kubernetes Contributor Assistant. Ask a question in English, or type 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in ("exit", "quit"):
            break
        if question:
            print_response(run(question, retriever, config))
    print("Bye!")


def main():
    parser = argparse.ArgumentParser(description="Kubernetes Contributor Assistant (RAG)")
    parser.add_argument("question", nargs="?", help="question in English; without it starts a dialog")
    parser.add_argument("--json", metavar="PATH", help="save the structured answer to a JSON file")
    parser.add_argument("--quiet", action="store_true", help="do not print the answer")
    args = parser.parse_args()

    config = load_config()["generation"]
    retriever = Retriever()

    if args.question is None:
        interactive(retriever, config)
        return

    response = run(args.question, retriever, config)
    if not args.quiet:
        print_response(response)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()