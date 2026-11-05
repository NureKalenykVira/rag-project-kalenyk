"""Behavioral tests (CheckList): MFT, INV and DIR. They call the real system, so they are slow."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import run
from retrieval import Retriever, load_config


RESULTS_PATH = Path(__file__).resolve().parent.parent / "reports" / "lab-5" / "behavior-results.json"


@pytest.fixture(scope="module")
def ask():
    config = load_config()["generation"]
    retriever = Retriever()
    cache = {}

    def _ask(question):
        # Same question is asked only once per test run
        if question not in cache:
            cache[question] = run(question, retriever, config)
        return cache[question]

    yield _ask
    # Save all answers so failed tests can be analysed in the report
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def mentions(response, *words):
    text = response["answer"].lower()
    return any(word.lower() in text for word in words)


def cited_docs(response):
    return {c["doc_id"] for c in response["citations"]}


def test_mft_cla(ask):
    r = ask("Do I need to sign a CLA before my first pull request?")
    assert r["status"] == "ok"
    assert mentions(r, "CLA", "Contributor License Agreement")
    assert "CLA.md" in cited_docs(r)


def test_mft_retest(ask):
    r = ask("How do I rerun failed tests on my pull request?")
    assert r["status"] == "ok"
    assert mentions(r, "/retest")


def test_inv_hold(ask):
    a = ask("How do I stop my PR from being merged too early?")
    b = ask("How can I prevent my pull request from being merged before it is ready?")
    assert mentions(a, "/hold") and mentions(b, "/hold")


def test_inv_squash(ask):
    a = ask("How do I squash my commits?")
    b = ask("How can I combine several commits into one before merging?")
    assert mentions(a, "rebase") and mentions(b, "rebase")


def test_dir_out_of_scope(ask):
    in_scope = ask("How do I sign the CLA?")
    out_of_scope = ask("How do I sign a lease for an apartment in Kyiv?")
    assert in_scope["status"] == "ok"
    assert out_of_scope["status"] == "refusal"


def test_dir_member_to_reviewer(ask):
    member = ask("What are the requirements to become a Kubernetes org member?")
    reviewer = ask("What are the requirements to become a reviewer?")
    assert mentions(member, "two-factor", "2FA")
    assert mentions(reviewer, "20", "5 PRs", "3 months")