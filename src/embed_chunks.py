"""Encode corpus chunks into vectors with a sentence-transformers model."""

import json
import time
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["embedding"]


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_texts(chunks, add_title):
    if add_title:
        return [f"{c['title']}\n{c['text']}" for c in chunks]
    return [c["text"] for c in chunks]


def count_truncated(model, texts):
    # The model silently cuts everything longer than max_seq_length tokens
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    truncated = sum(n > model.max_seq_length for n in lengths)
    return truncated, sum(lengths) / len(lengths)


def main():
    config = load_config()
    chunks = load_chunks(ROOT / config["chunks_path"])
    texts = build_texts(chunks, config["add_title"])

    model = SentenceTransformer(config["model_name"])
    model.max_seq_length = config["max_seq_length"]
    truncated, avg_tokens = count_truncated(model, texts)

    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=config["batch_size"],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    chunk_ids = [c["chunk_id"] for c in chunks]
    (output_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

    print(f"Model:            {config['model_name']}")
    print(f"Vector size:      {model.get_embedding_dimension()}")
    print(f"Max tokens:       {model.max_seq_length}")
    print(f"Chunks encoded:   {len(chunks)}")
    print(f"Avg tokens/chunk: {avg_tokens:.1f}")
    print(f"Truncated chunks: {truncated} ({truncated / len(chunks):.1%})")
    print(f"Encoding time:    {elapsed:.1f} s")
    print(f"Saved:            {embeddings.shape} -> {output_dir}")


if __name__ == "__main__":
    main()