"""Build a persistent Chroma index from precomputed chunk embeddings."""

import json
from pathlib import Path

import chromadb
import numpy as np
import yaml
import shutil


ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 500


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["index"]


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def folder_size_mb(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def main():
    config = load_config()
    chunks = load_chunks(ROOT / config["chunks_path"])
    embeddings_dir = ROOT / config["embeddings_dir"]
    embeddings = np.load(embeddings_dir / "embeddings.npy")
    chunk_ids = json.loads((embeddings_dir / "chunk_ids.json").read_text(encoding="utf-8"))

    # Vectors must be in the same order as chunks, otherwise metadata gets mixed up
    assert chunk_ids == [c["chunk_id"] for c in chunks], "Chunks and embeddings are out of sync"

    index_path = ROOT / config["path"]
    # Chroma does not free disk space after delete_collection, so start from an empty folder
    if index_path.exists():
        shutil.rmtree(index_path)
    client = chromadb.PersistentClient(path=str(index_path))
    
    collection = client.create_collection(
        name=config["collection"],
        embedding_function=None,
        configuration={"hnsw": {"space": config["space"], "ef_search": config["ef_search"]}},
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[start:start + BATCH_SIZE].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "title": c["title"],
                "n_words": c["n_words"],
            } for c in batch],
        )

    print(f"Collection:     {config['collection']} ({config['space']})")
    print(f"Vectors:        {collection.count()}")
    print(f"Dimension:      {embeddings.shape[1]}")
    print(f"Index on disk:  {folder_size_mb(index_path):.2f} MB -> {index_path}")
    print(f"HNSW ef_search: {config['ef_search']}")


if __name__ == "__main__":
    main()