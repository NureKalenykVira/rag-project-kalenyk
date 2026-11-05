"""Retrieval for the RAG system: dense (Chroma), sparse (BM25) and hybrid (RRF) search."""

import argparse
import json
import re
from pathlib import Path

import chromadb
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder, SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
TOKEN = re.compile(r"[a-z0-9]+")


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text):
    return TOKEN.findall(text.lower())


class Retriever:
    def __init__(self):
        config = load_config()
        self.model = SentenceTransformer(config["embedding"]["model_name"])
        self.model.max_seq_length = config["embedding"]["max_seq_length"]
        client = chromadb.PersistentClient(path=str(ROOT / config["index"]["path"]))
        self.collection = client.get_collection(config["index"]["collection"])

        with open(ROOT / config["index"]["chunks_path"], encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.bm25 = BM25Okapi([tokenize(f"{c['title']}\n{c['text']}") for c in self.chunks])

        self.rrf_k = config["retrieval"]["rrf_k"]
        self.candidates = config["retrieval"]["candidates"]
        self.reranker = CrossEncoder(config["retrieval"]["reranker"])
        self.rerank_candidates = config["retrieval"]["rerank_candidates"]
        self.max_chunks_per_doc = config["retrieval"]["max_chunks_per_doc"]
        
    def dense_search(self, query, k=5):
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        rows = zip(result["ids"][0], result["documents"][0],
                   result["metadatas"][0], result["distances"][0])
        for rank, (chunk_id, text, meta, distance) in enumerate(rows, start=1):
            hits.append({
                "chunk_id": chunk_id,
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "text": text,
                "score": 1 - distance,
                "rank": rank,
            })
        return hits

    def bm25_search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        best = np.argsort(scores)[::-1][:k]
        hits = []
        for rank, i in enumerate(best, start=1):
            chunk = self.chunks[i]
            hits.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "score": float(scores[i]),
                "rank": rank,
            })
        return hits

    def hybrid_search(self, query, k=5):
        results = {
            "dense": self.dense_search(query, self.candidates),
            "bm25": self.bm25_search(query, self.candidates),
        }
        fused = {}
        for method, hits in results.items():
            for hit in hits:
                if hit["chunk_id"] not in fused:
                    fused[hit["chunk_id"]] = {**hit, "score": 0.0,
                                              "dense_rank": None, "bm25_rank": None}
                entry = fused[hit["chunk_id"]]
                entry["score"] += 1 / (self.rrf_k + hit["rank"])
                entry[f"{method}_rank"] = hit["rank"]

        ranked = sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]
        for rank, hit in enumerate(ranked, start=1):
            hit["rank"] = rank
        return ranked


    def rerank_search(self, query, k=5):
        # A cross-encoder reads the query and the chunk together, so it judges relevance more precisely
        candidates = self.hybrid_search(query, self.rerank_candidates)
        pairs = [(query, f"{c['title']}\n{c['text']}") for c in candidates]
        scores = self.reranker.predict(pairs)
        for hit, score in zip(candidates, scores):
            hit["hybrid_rank"] = hit["rank"]
            hit["score"] = float(score)
        # Limit chunks per document so the prompt gets several different sources
        ranked, per_doc = [], {}
        for hit in sorted(candidates, key=lambda h: h["score"], reverse=True):
            if per_doc.get(hit["doc_id"], 0) < self.max_chunks_per_doc:
                ranked.append(hit)
                per_doc[hit["doc_id"]] = per_doc.get(hit["doc_id"], 0) + 1
            if len(ranked) == k:
                break
        for rank, hit in enumerate(ranked, start=1):
            hit["rank"] = rank
        return ranked
    

def print_hits(hits):
    for hit in hits:
        snippet = hit["text"][:150].replace("\n", " ")
        print(f"{hit['rank']}. [{hit['chunk_id']}]  score={hit['score']:.4f}")
        if "dense_rank" in hit:
            print(f"   dense rank: {hit['dense_rank'] or '-'}, bm25 rank: {hit['bm25_rank'] or '-'}")
        print(f"   {snippet}...\n")


def main():
    parser = argparse.ArgumentParser(description="Retrieval demo")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "rerank", "all"], default="hybrid")
    args = parser.parse_args()

    retriever = Retriever()
    searches = {"dense": retriever.dense_search, "bm25": retriever.bm25_search,
                "hybrid": retriever.hybrid_search,
                "rerank": retriever.rerank_search}
    for name, search in searches.items():
        if args.mode in (name, "all"):
            print(f"=== {name} search: {args.query}\n")
            print_hits(search(args.query, args.k))


if __name__ == "__main__":
    main()