ArxRAG: arXiv Multimodal Retrieval-Augmented Generation
======================================================

Overview
--------
ArxRAG is a lightweight Django 5 + DRF application that ingests arXiv PDFs (text + extracted page images), builds a FAISS vector index (OpenAI text-embedding-3-large, cosine via L2-normalized vectors), and serves a question‑answering API plus a minimal HTMX front‑end. Answers are produced using curated sentence snippets (not full chunk dumps) with source citation metadata.

Key Features
------------
- Text ingestion from arXiv PDFs (title, authors, per-page text chunking with overlap & dedup).
- Optional image (page region) extraction + CLIP embeddings (framework ready; retrieval currently text-focused).
- FAISS `IndexFlatIP` + embedding normalization (cosine similarity).
- Query pipeline with: keyword sentence scoring, numeric/table filtering, snippet selection, OpenAI chat completion with source citations.
- Frontend (HTMX) forms: ingest query + ask; collapsible context details (sources, snippets, token usage, latency, raw truncated chunks).
- Management command `reingest` to rebuild normalized index.

Architecture
------------
```
rag/ingest.py      -> fetch + chunk + embed + store + add to FAISS
rag/retrieval.py   -> search (FAISS) + answer (snippet assembly + LLM)
rag/agent.py       -> agent-style endpoints: /api/agent/search_ingest, /api/agent/ask
rag/views.py       -> basic /api/ask + home page view
rag/models.py      -> Document, Chunk, QueryLog
rag/mm.py          -> image extraction (optional)
templates/index.html -> HTMX UI
```

Endpoints
---------
| Method | Path                        | Body Example | Description |
|--------|-----------------------------|--------------|-------------|
| POST   | `/api/agent/search_ingest`  | `{ "query": "agentic RAG", "max_results": 1 }` | Fetch & ingest arXiv PDFs. |
| POST   | `/api/ask`                  | `{ "question": "How does MCP help RAG?", "k": 5 }` | Answer using existing corpus. |
| POST   | `/api/agent/ask`            | same as `/api/ask` | Agent namespace variant. |

Response (ask)
--------------
```
{
	"answer": "... synthesized answer ...",
	"meta": {
		"sources": [{"index":0,"paper":"Title","arxiv_id":"NNNN"...}],
		"snippets": ["[0] sentence ..."],
		"model": "gpt-4o",
		"usage": {"prompt_tokens":644,"completion_tokens":200,"total_tokens":844},
		"latency_s": 4.468,
		"context_token_counts": [245, 198, ...]
	},
	"contexts": [ {"kind":"text","content":"truncated chunk ...","ord":6}, ... ]
}
```

Prerequisites
-------------
- Conda environment named `djangoAI` (Python 3.11) with required libs.
- OpenAI API key stored at `~/.openai_api_key_gpt5` (single line).

Install (example)
-----------------
```
conda create -n djangoAI python=3.11 -y
conda activate djangoAI
pip install django djangorestframework faiss-cpu openai rapidfuzz pypdf pymupdf arxiv numpy
```

Run Migrations
--------------
```
conda activate djangoAI
python manage.py migrate
```

Ingest Example
--------------
```
conda activate djangoAI
python manage.py reingest --query "agentic RAG" --max-results 1
```

Dev Server
----------
```
conda activate djangoAI
python manage.py runserver 8000
```
Visit: http://127.0.0.1:8000/

Frontend Usage
--------------
1. Enter an arXiv search query (e.g., `agentic RAG`) and click Fetch & Index.
2. Ask a question. The answer card shows:
	 - Answer (top)
	 - Collapsible section (sources, snippets, model, latency, token usage, raw truncated chunks)

Design Choices
--------------
- Snippet-level grounding reduces hallucination while avoiding large context dumps.
- Numeric-heavy filtering avoids spending tokens on tables/metrics.
- L2 normalization enables cosine similarity with simple `IndexFlatIP`.
- Token counts (approx via whitespace) give lightweight visibility into context size.

Reingestion vs Reindex
----------------------
If embedding normalization logic changes, use `manage.py reingest` (will drop index & optionally data). For pure index rebuild from existing DB, you could implement a dedicated command to iterate chunks, normalize stored vectors, and rewrite FAISS (future enhancement).

Environment Notes
-----------------
Project-level `.vscode/settings.json` pins interpreter to the `djangoAI` env. All shell examples explicitly activate it.

Future Improvements
-------------------
- Sentence-level re-ranking using embedding similarity (current scoring = keyword overlap).
- Page number tracking for text chunks (store page in Chunk).
- Streaming answer support (Server-Sent Events or incremental HTMX swap).
- Rebuild-index command without re-downloading PDFs.
- Auth & rate limiting.
- CORS enablement for external frontend.

Troubleshooting
---------------
| Issue | Cause | Fix |
|-------|-------|-----|
| TemplateDoesNotExist `index.html` | Missing template DIRS path | Ensure `TEMPLATES.DIRS` contains `BASE_DIR / 'templates'` (already configured). |
| Raw JSON showed in UI | htmx swapped JSON | Frontend now cancels JSON swaps & renders manually. Hard refresh. |
| Duplicate retrieval rows | FAISS returning same index multiple times | Dedup added in `search()`. |
| Irrelevant numeric context | Table-like chunk | Numeric-heavy filter trims or skips. |

License
-------
Add a license of your choice (MIT, Apache 2.0, etc.).

Attribution
-----------
Uses OpenAI API for embeddings & chat completion; arXiv content per arXiv terms of use.

