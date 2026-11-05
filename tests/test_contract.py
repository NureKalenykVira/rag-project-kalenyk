"""Unit tests for the citation contract (no model needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contract import REFUSAL, check_answer, extract_citations
import generate


HITS = [{"chunk_id": f"doc.md#{i}", "doc_id": "doc.md", "title": "Doc", "text": "Some text."}
        for i in range(5)]


def fake_model(answers):
    calls = iter(answers)
    return lambda messages, config: next(calls)


def test_extract_formats():
    assert extract_citations("A [1]. B [2][3]. C [1, 4].") == [1, 2, 3, 4]


def test_ok():
    assert check_answer("Sign the CLA [3].", 5)["status"] == "ok"


def test_no_citations():
    assert check_answer("Sign the CLA.", 5)["status"] == "no_citations"


def test_invalid_citation():
    check = check_answer("Sign the CLA [7].", 5)
    assert check["status"] == "invalid_citations"
    assert check["invalid"] == [7]


def test_refusal():
    assert check_answer(REFUSAL, 5)["status"] == "refusal"


def test_mixed():
    assert check_answer(f"Enable 2FA [1]. {REFUSAL}", 5)["status"] == "mixed"


def test_retry_fixes_answer(monkeypatch):
    monkeypatch.setattr(generate, "ask_model", fake_model(["Sign the CLA.", "Sign the CLA [3]."]))
    result = generate.answer_question("q", HITS, {})
    assert result["first_status"] == "no_citations"
    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert result["cited_chunks"] == ["doc.md#2"]


def test_retry_fails_uncertain(monkeypatch):
    monkeypatch.setattr(generate, "ask_model", fake_model(["Sign the CLA [7].", "Sign the CLA [9]."]))
    result = generate.answer_question("q", HITS, {})
    assert result["first_status"] == "invalid_citations"
    assert result["status"] == "uncertain"
    assert result["attempts"] == 2


def test_refusal_no_retry(monkeypatch):
    monkeypatch.setattr(generate, "ask_model", fake_model([generate.REFUSAL]))
    result = generate.answer_question("q", HITS, {})
    assert result["status"] == "refusal"
    assert result["attempts"] == 1