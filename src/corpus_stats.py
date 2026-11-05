"""Count corpus statistics and top words for the data card."""

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "data" / "processed" / "docs.jsonl"
WORD = re.compile(r"[a-z][a-z0-9]+")

STOP_WORDS = {
    "the", "to", "and", "of", "a", "in", "is", "for", "be", "on", "or", "that",
    "this", "with", "are", "as", "it", "an", "by", "if", "at", "can", "should",
    "will", "from", "not", "have", "has", "you", "your", "we", "our", "they",
    "their", "any", "all", "more", "when", "which", "who", "what", "how", "there",
    "these", "those", "also", "may", "must", "but", "about", "into", "other",
    "than", "then", "so", "do", "does", "been", "was", "were", "would", "could",
    "one", "each", "such", "its", "them", "only", "some", "no", "use", "used",
    "see", "via", "e", "g", "i", "s", "etc", "us", "out", "up", "new",
}


def main():
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    words = [w for doc in docs for w in WORD.findall(doc["text"].lower())]
    counts = Counter(words)
    content_counts = Counter({w: c for w, c in counts.items() if w not in STOP_WORDS})

    print(f"Documents: {len(docs)}")
    print(f"Words:     {len(words)}")
    print(f"Unique:    {len(counts)}")

    print("\nTop 20 (all words):")
    for word, count in counts.most_common(20):
        print(f"| {word} | {count} |")

    print("\nTop 20 (without stop words):")
    for word, count in content_counts.most_common(20):
        print(f"| {word} | {count} |")


if __name__ == "__main__":
    main()