"""Split cleaned documents into overlapping fixed-size chunks."""

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
WORD = re.compile(r"\S+")


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["chunking"]


def load_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_into_chunks(text, size, overlap):
    words = list(WORD.finditer(text))
    step = size - overlap
    chunks = []
    start = 0
    while True:
        end = start + size
        if len(words) - end < overlap:
            end = len(words)
        window = words[start:end]
        chunks.append((text[window[0].start():window[-1].end()], len(window)))
        if end >= len(words):
            break
        start += step
    return chunks


def main():
    config = load_config()
    docs = load_docs(ROOT / config["input_path"])

    records = []
    for doc in docs:
        chunks = split_into_chunks(doc["text"], config["chunk_size"], config["overlap"])
        for index, (text, n_words) in enumerate(chunks):
            records.append({
                "chunk_id": f"{doc['doc_id']}#{index}",
                "doc_id": doc["doc_id"],
                "chunk_index": index,
                "title": doc["title"],
                "text": text,
                "n_words": n_words,
            })

    output_path = ROOT / config["output_path"]
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    sizes = [r["n_words"] for r in records]
    print(f"Documents:        {len(docs)}")
    print(f"Chunks:           {len(records)}")
    print(f"Avg chunk size:   {sum(sizes) / len(sizes):.1f} words")
    print(f"Min / max size:   {min(sizes)} / {max(sizes)} words")
    print(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()