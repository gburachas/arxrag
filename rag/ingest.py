import os, io, arxiv, numpy as np
from pypdf import PdfReader
from rapidfuzz.distance import Levenshtein
from django.conf import settings
from .models import Document, Chunk, Reference
import faiss

INDEX_PATH = "data/index/faiss_text.index"
REF_INDEX_PATH = "data/index/faiss_refs.index"
DIM = 3072  # match your embedder

def get_embedder():
    """Return a callable that embeds a list of texts while respecting model token limits.

    The OpenAI text-embedding-3-large model has an 8192 token context window PER request.
    We (roughly) treat whitespace-delimited words as tokens (fast, conservative) so we keep
    each item <= 8190 "tokens" and batch groups so their combined size does not exceed a
    safety threshold (default 6000) to avoid 400 errors.
    """
    # Offline / test mode: if env var set or key file missing, return deterministic dummy embedder
    key_path = os.path.expanduser("~/.openai_api_key_gpt5")
    offline = os.environ.get("RAG_OFFLINE_TEST") == "1" or not os.path.exists(key_path)
    client = None
    if not offline:
        try:
            with open(key_path) as f:
                api_key = f.read().strip()
            from openai import OpenAI  # only import when actually used
            client = OpenAI(api_key=api_key)
        except Exception:
            # Fallback to offline mode if anything goes wrong
            offline = True

    MODEL = "text-embedding-3-large"
    PER_ITEM_LIMIT = 8190  # safety (model is 8192)
    BATCH_LIMIT = 6000     # approx total tokens per batch (heuristic)

    def _truncate(t: str):
        toks = t.split()
        if len(toks) > PER_ITEM_LIMIT:
            toks = toks[:PER_ITEM_LIMIT]
        return " ".join(toks)

    rng = np.random.default_rng(12345)

    def embed(texts):
        if offline:
            # Deterministic pseudo-embedding: hash to RNG seed + Gaussian vector
            vecs = []
            for t in texts:
                h = abs(hash(t)) % (2**32 - 1)
                local_rng = np.random.default_rng(h)
                v = local_rng.normal(size=(DIM,)).astype('float32')
                n = np.linalg.norm(v)
                if n > 0:
                    v /= n
                vecs.append(v)
            out = np.stack(vecs, axis=0)
            assert out.shape[1] == DIM, f"Offline embedding produced wrong dim {out.shape[1]} != {DIM}"
            return out
        # Real embedding path
        all_vecs = []
        batch = []
        batch_tok_count = 0
        for raw in texts:
            t = _truncate(raw)
            t_tok = len(t.split())
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
        arr = np.array(all_vecs, dtype="float32")
        if arr.shape[1] != DIM:
            raise ValueError(f"Embedding dimension mismatch: got {arr.shape[1]}, expected {DIM}")
        return arr

    return embed

def load_or_new_index(d=DIM):
    """Load existing FAISS chunk index or create a new normalized IP index.

    NOTE: The ingestion bug we observed was an index on disk containing only 1 vector
    while the database had ~700 chunks. Root cause: index file was overwritten early
    (e.g. via partial ingest) and not rebuilt after additional chunks were added via
    direct DB population (tests or migrations) or a previous interrupted run.

    We keep behavior simple here (just load if present) and provide an explicit
    rebuild management command elsewhere. This function intentionally does NOT try
    to reconcile DB vs index sizes implicitly (could be expensive on each ingest).
    """
    if os.path.exists(INDEX_PATH):
        try:
            return faiss.read_index(INDEX_PATH)
        except Exception:
            # Corrupt index -> start fresh (caller will add vectors)
            pass
    return faiss.IndexFlatIP(d)

def load_or_new_ref_index(d=DIM):
    if os.path.exists(REF_INDEX_PATH):
        return faiss.read_index(REF_INDEX_PATH)
    return faiss.IndexFlatIP(d)

def save_index(idx): faiss.write_index(idx, INDEX_PATH)
def save_ref_index(idx): faiss.write_index(idx, REF_INDEX_PATH)

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

def _extract_references(text_pages):
    """Reference extraction using regex segmentation of bracketed or numbered citations.

    Strategy:
      1. Scan last few pages for a heading containing 'references' or 'bibliography'.
      2. Remove the heading line.
      3. Join page text; segment with regex pattern capturing leading [n], (n), or n. tokens.
      4. Clean & deduplicate.
    """
    import re
    candidate_text = []
    for p in text_pages[-6:]:
        if not p:
            continue
        lower = p.lower()
        if "references" in lower or "bibliography" in lower:
            # trim everything before the heading word
            m = re.search(r'(references|bibliography)', lower)
            if m:
                p = p[m.start():]
            candidate_text.append(p)
    if not candidate_text:
        return []
    joined = "\n".join(candidate_text)
    # Remove header tokens
    joined = re.sub(r'^(references|bibliography)\s*', '', joined, flags=re.I)
    # Split by patterns starting a new reference
    # Pattern: start of line, optional spaces, then ([n]) or [n] or n. or n) followed by space
    pattern = re.compile(r'(?m)^(?=\s*(?:\[[0-9]+\]|\([0-9]+\)|[0-9]+[.)]))')
    raw_refs = [r.strip() for r in pattern.split(joined) if r.strip()]
    cleaned = []
    for r in raw_refs:
        # merge internal newlines
        r = re.sub(r'\s+', ' ', r).strip()
        # discard obvious non-reference noise
        if len(r.split()) < 4:
            continue
        # must have a year-like pattern or a period for plausible reference
        if not re.search(r'(19|20)\d{2}', r) and '.' not in r:
            continue
        cleaned.append(r)
    # deduplicate on first 120 chars
    uniq = []
    seen = set()
    for r in cleaned:
        key = r[:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq

def ingest_arxiv(query="agentic RAG", max_results=1, embed_references=True):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    embed = get_embedder()
    idx = load_or_new_index()
    ref_idx = load_or_new_ref_index()

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

        # References
        if embed_references:
            refs = _extract_references(text_pages)
            if refs:
                ref_vecs = embed(refs)
                # normalize
                norms = np.linalg.norm(ref_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                ref_vecs = ref_vecs / norms
                import re
                arxiv_pattern = re.compile(r'arXiv[:\s]?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)', re.I)
                author_split_pattern = re.compile(r'(?:(?:\b[A-Z][a-zA-Z\-]+\s+(?:[A-Z]\.)+)|(?:[A-Z]\.\s*[A-Z][a-zA-Z\-]+))')
                name_pattern = re.compile(r"\b([A-Z][a-zA-Z\-']+(?:\s+[A-Z]\.)+(?:\s+[A-Z][a-zA-Z\-']+)?|[A-Z]\.\s*[A-Z][a-zA-Z\-']+)\b")
                stop_words = {"and", "et", "al", "In", "Proceedings", "Journal", "IEEE", "ACM"}
                for i, (t, v) in enumerate(zip(refs, ref_vecs)):
                    arxiv_id = ""
                    m = arxiv_pattern.search(t)
                    if m:
                        arxiv_id = m.group(1)
                    year_match = re.search(r'(19|20)\d{2}', t)
                    year_pos = year_match.start() if year_match else 180
                    candidate_segment = t[:year_pos]
                    candidates = []
                    for mname in name_pattern.finditer(candidate_segment):
                        name = mname.group(0).strip().strip(',;')
                        if any(part in stop_words for part in name.split()):
                            continue
                        # basic quality gates
                        if len(name) < 3 or sum(ch.isalpha() for ch in name) < 3:
                            continue
                        candidates.append(name)
                    # de-duplicate preserving order
                    seen_names = set()
                    authors_list = []
                    for nm in candidates:
                        key = nm.lower()
                        if key in seen_names:
                            continue
                        seen_names.add(key)
                        authors_list.append(nm)
                        if len(authors_list) >= 6:
                            break
                    authors = ", ".join(authors_list)
                    Reference.objects.create(doc=doc, raw_text=t, position=i, vector=v.tobytes(), arxiv_id=arxiv_id, authors=authors)
                ref_idx.add(ref_vecs)
    # Persist updated chunk index
    save_index(idx)
    if embed_references:
        save_ref_index(ref_idx)
