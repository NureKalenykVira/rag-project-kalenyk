"""Search the Chroma index: print the top-k chunks for a query."""

import argparse
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_search():
    config = load_config()
    # Must be encoded exactly like the chunks
    model = SentenceTransformer(config["embedding"]["model_name"])
    model.max_seq_length = config["embedding"]["max_seq_length"]

    client = chromadb.PersistentClient(path=str(ROOT / config["index"]["path"]))
    collection = client.get_collection(config["index"]["collection"])
    return model, collection


def search(model, collection, query, k=5):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                              result["metadatas"][0], result["distances"][0]):
        hits.append({"chunk_id": chunk_id, "title": meta["title"],
                     "text": text, "similarity": 1 - distance})
    return hits


def main():
    parser = argparse.ArgumentParser(description="Search the chunk index")
    parser.add_argument("query", help="question in English")
    parser.add_argument("--k", type=int, default=5, help="number of results")
    args = parser.parse_args()

    model, collection = load_search()
    hits = search(model, collection, args.query, args.k)

    print(f"Query: {args.query}\n")
    for rank, hit in enumerate(hits, start=1):
        snippet = hit["text"][:200].replace("\n", " ")
        print(f"{rank}. [{hit['chunk_id']}]  similarity={hit['similarity']:.3f}")
        print(f"   {hit['title']}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()