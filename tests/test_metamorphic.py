"""Metamorphic tests: a paraphrased query should find and cite the same sources."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag import run
from retrieval import Retriever, load_config


GOLD_PATH = ROOT / "tests" / "gold-set.jsonl"
RESULTS_PATH = ROOT / "reports" / "lab-5" / "metamorphic-results.json"

PARAPHRASES = {
    "q01": "Where can I find beginner-friendly tasks to start contributing?",
    "q03": "What do I need to do to join the Kubernetes GitHub organization?",
    "q06": "Someone behaved badly towards me in the community. Who do I tell?",
    "q07": "Is there an agreement I have to accept before contributing code?",
    "q08": "The checks on my PR failed. How can I trigger them again?",
    "q10": "I found a bug that could be exploited by attackers. Who should I contact?",
    "q12": "How should I describe my changes when I commit them?",
    "q13": "What is the purpose of the OWNERS files in Kubernetes repositories?",
    "q14": "How can I get reviewer rights in a Kubernetes repository?",
    "q18": "What are the guidelines for chatting in the Kubernetes Slack workspace?",
}

GENERATION_IDS = ["q06", "q07", "q10", "q13"]

MIN_RECALL_AGREEMENT = 0.7
MIN_RETRIEVAL_JACCARD = 0.3
MIN_CITED_DOC_JACCARD = 0.3


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


@pytest.fixture(scope="module")
def gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    return {item["id"]: item for item in items}


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@pytest.fixture(scope="module")
def results():
    data = {}
    yield data
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def retrieved(gold, retriever):
    k = load_config()["generation"]["top_k"]
    pairs = {}
    for qid, paraphrase in PARAPHRASES.items():
        original = retriever.rerank_search(gold[qid]["query"], k)
        rephrased = retriever.rerank_search(paraphrase, k)
        pairs[qid] = ([h["chunk_id"] for h in original], [h["chunk_id"] for h in rephrased])
    return pairs


def test_recall_invariance(gold, retrieved, results):
    rows = []
    for qid, (original, rephrased) in retrieved.items():
        relevant = set(gold[qid]["relevant_chunk_ids"])
        rows.append({
            "id": qid,
            "original_hit": bool(relevant & set(original)),
            "paraphrase_hit": bool(relevant & set(rephrased)),
        })
    agreement = sum(r["original_hit"] == r["paraphrase_hit"] for r in rows) / len(rows)
    results["recall_invariance"] = {
        "agreement": round(agreement, 2),
        "recall_original": round(sum(r["original_hit"] for r in rows) / len(rows), 2),
        "recall_paraphrase": round(sum(r["paraphrase_hit"] for r in rows) / len(rows), 2),
        "pairs": rows,
    }
    assert agreement >= MIN_RECALL_AGREEMENT


def test_retrieval_overlap(retrieved, results):
    rows = [
        {"id": qid, "jaccard": round(jaccard(original, rephrased), 2)}
        for qid, (original, rephrased) in retrieved.items()
    ]
    mean = sum(r["jaccard"] for r in rows) / len(rows)
    results["retrieval_overlap"] = {"mean_jaccard": round(mean, 2), "pairs": rows}
    assert mean >= MIN_RETRIEVAL_JACCARD


def test_citation_overlap(gold, retriever, results):
    config = load_config()["generation"]
    rows = []
    for qid in GENERATION_IDS:
        a = run(gold[qid]["query"], retriever, config)
        b = run(PARAPHRASES[qid], retriever, config)
        rows.append({
            "id": qid,
            "statuses": [a["status"], b["status"]],
            "chunk_jaccard": round(jaccard([c["chunk_id"] for c in a["citations"]],
                                           [c["chunk_id"] for c in b["citations"]]), 2),
            "doc_jaccard": round(jaccard([c["doc_id"] for c in a["citations"]],
                                         [c["doc_id"] for c in b["citations"]]), 2),
            "answers": [a["answer"], b["answer"]],
        })
    mean_chunk = sum(r["chunk_jaccard"] for r in rows) / len(rows)
    mean_doc = sum(r["doc_jaccard"] for r in rows) / len(rows)
    results["citation_overlap"] = {
        "mean_chunk_jaccard": round(mean_chunk, 2),
        "mean_doc_jaccard": round(mean_doc, 2),
        "pairs": rows,
    }
    assert mean_doc >= MIN_CITED_DOC_JACCARD
