"""Contract at the model/code boundary: parse and validate citations in the model answer."""

import re


REFUSAL = "I don't have enough information in the Kubernetes community documentation to answer this."
COMMUNITY_ADVICE = ("You can ask in the #kubernetes-contributors channel on the Kubernetes Slack "
                    "or on the dev@kubernetes.io mailing list.")
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


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