# Kubernetes Contributor Assistant

A RAG assistant that answers newcomer questions about contributing to Kubernetes
(pull requests, reviews, membership, SIGs, code of conduct) using the
kubernetes/community documentation, with citations to source documents.

Corpus: https://github.com/kubernetes/community (Apache-2.0), 535 documents, 2,772 chunks.

## Stack
- Python 3.11+
- Chroma (vector store)
- sentence-transformers: all-MiniLM-L6-v2 (embeddings), ms-marco-MiniLM-L-6-v2 (reranker)
- rank_bm25 (keyword search)
- Ollama: qwen2.5:7b (generation)

## How it works
1. Hybrid search: dense (Chroma) + BM25, merged with Reciprocal Rank Fusion (k = 60).
2. Reranking of the top 20 candidates with a cross-encoder, at most 2 chunks per document.
3. The top 5 chunks are passed to the model, which answers with citations like [1].
4. A citation contract checks the answer; on a violation the model gets one retry,
otherwise the answer is marked as uncertain. Out-of-scope questions are refused.

## Installation
1. Install Python 3.11+ and Ollama (https://ollama.com).
2. `ollama pull qwen2.5:7b`
3. `python -m venv .venv`
4. `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/macOS)
5. `python -m pip install -r requirements.txt`

## Building the index
The processed corpus and the index are already in the repository. To rebuild them from scratch:
```
python src/fetch_corpus.py
python src/clean_corpus.py
python src/chunk_corpus.py
python src/embed_chunks.py
python src/build_index.py
```

## Usage
Interactive dialog:
```
python src/rag.py
```
Single question, optionally saved as JSON:
```
python src/rag.py "Do I need to sign anything before my first pull request?"
python src/rag.py "How do I squash my commits?" --json answer.json
```

## Evaluation
```
python src/evaluate_retrieval.py
python -m pytest tests -v
```