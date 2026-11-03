## Kubernetes Contributor Assistant

A RAG assistant that answers newcomer questions about contributing to Kubernetes
(pull requests, reviews, membership, SIGs, code of conduct) using the
kubernetes/community documentation, with citations to source documents.

Corpus: https://github.com/kubernetes/community (Apache-2.0)

### Stack
- Python 3.11+
- Chroma (vector store)
- sentence-transformers, all-MiniLM-L6-v2 (embeddings)
- rank_bm25 (keyword search)
- Ollama, qwen2.5:3b (generation)

### Installation
1. Install Python 3.11+ and Ollama (https://ollama.com).
2. Pull the model: `ollama pull qwen2.5:3b`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/macOS)
5. Install dependencies: `pip install chromadb sentence-transformers rank_bm25 ollama`

Usage instructions will be added.