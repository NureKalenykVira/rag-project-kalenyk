"""Contract at the model/code boundary: parse and validate citations, block prompt injections."""

import re


REFUSAL = "I don't have enough information in the Kubernetes community documentation to answer this."
COMMUNITY_ADVICE = ("You can ask in the #kubernetes-contributors channel on the Kubernetes Slack "
                    "or on the dev@kubernetes.io mailing list.")
BLOCKED = ("I can only answer questions about contributing to Kubernetes, "
           "in English, with citations from the community documentation.")
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

INJECTION = re.compile(
    r"ignore\s+(all\s+|any\s+|your\s+|the\s+)*(previous\s+|prior\s+|above\s+)?(instructions|rules|prompt)"
    r"|system\s+prompt"
    r"|you\s+are\s+now"
    r"|pretend\s+(to\s+be|you\s+are)"
    r"|(do\s+not|don't)\s+cite"
    r"|ігноруй"
    r"|системн\w*\s+промпт",
    re.IGNORECASE,
)


def extract_citations(answer):
    numbers = set()
    for group in CITATION.findall(answer):
        numbers.update(int(n) for n in group.split(","))
    return sorted(numbers)


def check_answer(answer, n_sources):
    cited = extract_citations(answer)
    invalid = [n for n in cited if not 1 <= n <= n_sources]
    refusal = REFUSAL.lower() in answer.lower()

    if refusal and not cited:
        status = "refusal"
    elif refusal:
        status = "mixed"
    elif not cited:
        status = "no_citations"
    elif invalid:
        status = "invalid_citations"
    else:
        status = "ok"
    return {"status": status, "cited": cited, "invalid": invalid}


def make_feedback(check, n_sources):
    # Concrete external feedback: the model is told exactly what was wrong
    if check["status"] == "no_citations":
        return "Your answer has no citations. Put the source number like [1] after every statement."
    if check["status"] == "invalid_citations":
        return (f"Your answer cites sources that do not exist: {check['invalid']}. "
                f"Only sources 1-{n_sources} are available. Fix the citations.")
    return ("Your answer both answers the question and says there is not enough information. "
            f'Either answer with citations, or reply only: "{REFUSAL}"')


def looks_like_injection(question):
    return INJECTION.search(question) is not None


def is_english(answer, max_share=0.05):
    letters = [ch for ch in answer if ch.isalpha()]
    if not letters:
        return True
    foreign = sum(1 for ch in letters if not ch.isascii())
    return foreign / len(letters) <= max_share