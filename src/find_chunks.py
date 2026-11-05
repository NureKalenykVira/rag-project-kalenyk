"""Find chunks containing all given keywords (helper for building the eval set)."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
MAX_RESULTS = 15


def main():
    args = sys.argv[1:]
    doc_filter = None
    if "--doc" in args:
        i = args.index("--doc")
        doc_filter = args[i + 1]
        args = args[:i] + args[i + 2:]

    keywords = [k.lower() for k in args]
    if not keywords:
        print('Usage: python src/find_chunks.py "keyword" ["keyword" ...] [--doc path]')
        return

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    found = [c for c in chunks
             if all(k in c["text"].lower() for k in keywords)
             and (doc_filter is None or doc_filter in c["doc_id"])]
    print(f"Found {len(found)} chunks\n")
    for chunk in found[:MAX_RESULTS]:
        text = chunk["text"]
        pos = text.lower().find(keywords[0])
        snippet = text[max(0, pos - 150):pos + 250].replace("\n", " ")
        print(f"[{chunk['chunk_id']}] {chunk['title']}")
        print(f"  ...{snippet}...\n")


if __name__ == "__main__":
    main()