import os, io, arxiv, numpy as np
from pypdf import PdfReader
from rapidfuzz.distance import Levenshtein
from django.conf import settings
from .models import Document, Chunk
import faiss

INDEX_PATH = "data/index/faiss_text.index"
DIM = 3072  # match your embedder

def get_embedder():
    """Return a callable that embeds a list of texts while respecting model token limits.

    The OpenAI text-embedding-3-large model has an 8192 token context window PER request.
    We (roughly) treat whitespace-delimited words as tokens (fast, conservative) so we keep
    each item <= 8190 "tokens" and batch groups so their combined size does not exceed a
    safety threshold (default 6000) to avoid 400 errors.
    """
    with open(os.path.expanduser("~/.openai_api_key_gpt5")) as f:
        api_key = f.read().strip()
    print("API KEY:", repr(api_key))  # debug (consider removing in prod)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    MODEL = "text-embedding-3-large"
    PER_ITEM_LIMIT = 8190  # safety (model is 8192)
    BATCH_LIMIT = 6000     # approx total tokens per batch (heuristic)

    def _truncate(t: str):
        toks = t.split()
        if len(toks) > PER_ITEM_LIMIT:
            toks = toks[:PER_ITEM_LIMIT]
        return " ".join(toks)

    def embed(texts):
        all_vecs = []
        batch = []
        batch_tok_count = 0
        for raw in texts:
            t = _truncate(raw)
            t_tok = len(t.split())
            # flush if adding would overflow batch heuristic
            if batch and batch_tok_count + t_tok > BATCH_LIMIT:
                out = client.embeddings.create(model=MODEL, input=batch)
                all_vecs.extend(d.embedding for d in out.data)
                batch = []
                batch_tok_count = 0
            batch.append(t)
            batch_tok_count += t_tok
        if batch:
            out = client.embeddings.create(model=MODEL, input=batch)
            all_vecs.extend(d.embedding for d in out.data)
        return np.array(all_vecs, dtype="float32")

    return embed

def load_or_new_index(d=DIM):
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    idx = faiss.IndexFlatIP(d); return idx

def save_index(idx): faiss.write_index(idx, INDEX_PATH)

def chunk_text(pages, max_tokens=350, overlap=60):
    """Create semi-overlapping chunks constrained to max_tokens (approx words).

    We strictly enforce that each emitted chunk <= max_tokens by trimming instead of
    letting chunks overshoot then resetting. Uses simple whitespace tokenization.
    """
    chunks = []
    window = []
    window_tokens = 0
    for p in pages:
        toks = p.split()
        for tok in toks:
            window.append(tok)
            window_tokens += 1
            if window_tokens >= max_tokens:
                chunks.append(" ".join(window))
                # start new window with overlap
                if overlap > 0:
                    window = window[-overlap:]
                    window_tokens = len(window)
                else:
                    window = []
                    window_tokens = 0
        # continue accumulating
    if window_tokens > 0:
        chunks.append(" ".join(window))
    # simple dedup using Levenshtein distance threshold
    dedup = []
    for c in chunks:
        if not dedup or Levenshtein.distance(c, dedup[-1]) > 50:
            dedup.append(c)
    return dedup

def ingest_arxiv(query="agentic RAG", max_results=1):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    embed = get_embedder()
    idx = load_or_new_index()

    for r in search.results():
        pdf_path = f"data/pdfs/{r.get_short_id()}.pdf"
        if not os.path.exists(pdf_path):
            r.download_pdf(filename=pdf_path)
        doc = Document.objects.create(
            arxiv_id=r.get_short_id(),
            title=r.title,
            authors=", ".join(a.name for a in r.authors),
            pdf_path=pdf_path,
        )
        # Extract plain text per page
        reader = PdfReader(pdf_path)
        text_pages = [(page.extract_text() or "") for page in reader.pages]
        parts = chunk_text(text_pages)
        if not parts:
            continue
        vecs = embed(parts)
        # L2 normalize (cosine similarity with IndexFlatIP)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        print("Embedding shape:", vecs.shape)
        # Persist each chunk
        for i, (t, v) in enumerate(zip(parts, vecs)):
            Chunk.objects.create(
                doc=doc,
                kind="text",
                content=t,
                ord=i,
                vector=v.tobytes(),
            )
        # Add vectors to FAISS index
        idx.add(vecs)
    save_index(idx)
