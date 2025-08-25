import numpy as np, faiss, os, time
from .models import Chunk
from .ingest import get_embedder, INDEX_PATH, DIM

def search(q, k=5, multimodal=True):
    embed = get_embedder()
    qv = embed([q]).astype("float32")
    # Normalize query vector to match normalized index vectors (cosine similarity via inner product)
    q_norm = np.linalg.norm(qv, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    qv = qv / q_norm
    idx = faiss.read_index(INDEX_PATH)
    D, I = idx.search(qv, k)
    chunks = Chunk.objects.all().order_by("id")
    hits = []
    seen = set()
    for row in I[0]:
        idx_py = int(row)  # Convert numpy.int64 to Python int
        if 0 <= idx_py < chunks.count():
            if idx_py in seen:
                continue
            seen.add(idx_py)
            ch = chunks[idx_py]
            hits.append(ch)
        else:
            # Optionally log or skip out-of-bounds index
            pass
    return hits

def answer1(q, k=5):
    ctxs = search(q, k)
    context_text = "\n\n".join(f"[{i}] {c.content[:800]}" for i,c in enumerate(ctxs) if c.kind=="text")
    from openai import OpenAI
    import time

        # Read API key from file
    with open(os.path.expanduser("~/.openai_api_key_gpt5")) as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)

    msg = [
      {"role":"system","content":"You are a scholarly assistant. Cite brackets [i] from context."},
      {"role":"user","content":f"Question: {q}\n\nContext:\n{context_text}\n\nAnswer:"}
    ]
    out = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.2)
    return out.choices[0].message.content, ctxs

def answer(q, k=5):
    print("answer called with:", q, k)
    original_ctxs = search(q, k)
    # --- Redundancy suppression: collapse near-identical chunks (exact hash match) ---
    import hashlib
    seen_hash = {}
    dedup_ctxs = []
    old_to_new = {}
    for idx, c in enumerate(original_ctxs):
        text = (c.content or "")
        h = hashlib.sha1(text.encode('utf-8')).hexdigest()
        if h in seen_hash:
            # duplicate: map to existing canonical index
            old_to_new[idx] = seen_hash[h]
            continue
        new_index = len(dedup_ctxs)
        seen_hash[h] = new_index
        old_to_new[idx] = new_index
        dedup_ctxs.append(c)
    ctxs = dedup_ctxs
    print("search returned count:", len(ctxs))

    def is_numeric_heavy(text: str) -> bool:
        if not text:
            return True
        tokens = text.split()
        if not tokens:
            return True
        digitish = sum(1 for t in tokens if sum(ch.isdigit() for ch in t) >= max(1, len(t)//2))
        return digitish / max(1, len(tokens)) > 0.45  # skip tables/metrics

    # Score chunks by simple keyword overlap (lowercase unique words from question)
    q_words = {w for w in q.lower().split() if len(w) > 2}
    scored = []
    for old_i, c in enumerate(ctxs):
        if c.kind != "text":
            continue
        txt = c.content.strip()
        if not txt:
            continue
        if is_numeric_heavy(txt):
            # attempt to salvage first non-numeric sentence
            parts = [p.strip() for p in txt.split('.') if p.strip()]
            for p in parts:
                if not is_numeric_heavy(p):
                    txt = p
                    break
            else:
                continue  # give up if all numeric heavy
        words = [w.strip('.,();:') for w in txt.lower().split()]
        overlap = len(q_words.intersection(words))
    # use current (deduped) index
    scored.append((overlap, old_i, c, txt))

    # sort by score desc, then index
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Extract *relevant sentences* (not whole chunks) to give the model factual grounding
    # without dumping entire sections or tables.
    import re
    sent_split = re.compile(r'(?<=[.!?])\s+')

    def clean_sentence(s: str) -> str:
        s = s.strip()
        # remove excessive whitespace
        s = re.sub(r'\s+', ' ', s)
        return s

    q_words = {w for w in q.lower().split() if len(w) > 2}
    sentence_records = []  # (score, chunk_index, sentence)
    for overlap, i, c, txt in scored[: k * 4]:  # look a bit deeper pool
        sentences = [clean_sentence(s) for s in sent_split.split(txt) if len(s.strip()) > 20]
        # score each sentence by overlap with question words
        for snt in sentences:
            words = {w.strip('.,();:') for w in snt.lower().split() if w}
            score = len(q_words.intersection(words))
            if score == 0:
                # allow some high-information sentences by fallback heuristic: contains a keyword-like capitalized term
                if any(ch.isupper() for ch in snt[:80]):
                    score = 1
            if score > 0 and not is_numeric_heavy(snt):
                sentence_records.append((score, i, snt))

    # sort sentences by score desc then chunk index
    sentence_records.sort(key=lambda x: (-x[0], x[1]))
    # Remove duplicate sentences (same lowercase text) keeping first (highest score) occurrence
    seen_sent = set()
    uniq_sentence_records = []
    for rec in sentence_records:
        key = rec[2].lower()
        if key in seen_sent:
            continue
        seen_sent.add(key)
        uniq_sentence_records.append(rec)
    sentence_records = uniq_sentence_records

    # Build concise snippet list with word budget
    total_word_budget = 800
    used_words = 0
    per_chunk_cap = 3
    per_chunk_counts = {}
    snippet_list = []  # entries like [i] sentence
    for score, i, snt in sentence_records:
        if used_words >= total_word_budget:
            break
        per_chunk_counts.setdefault(i, 0)
        if per_chunk_counts[i] >= per_chunk_cap:
            continue
        w = snt.split()
        if used_words + len(w) > total_word_budget:
            continue
        snippet_list.append(f"[{i}] {snt}")
        per_chunk_counts[i] += 1
        used_words += len(w)
    # fallback: if nothing gathered (e.g., very short question), include first sentence of top chunks
    if not snippet_list:
        for _, i, c, txt in scored[:k]:
            first = sent_split.split(txt)[0].strip()[:400]
            if first:
                snippet_list.append(f"[{i}] {first}")
    # Build source metadata lines (no raw content) for the model to cite.
    def extract_page(c):
        if c.kind == "image" and c.image_path:
            # pattern: ...pdf_<page>_<xref>.png
            base = os.path.basename(c.image_path)
            parts = base.split('_')
            if len(parts) >= 3 and parts[-1].endswith('.png'):
                try:
                    return int(parts[-2])
                except ValueError:
                    return None
        return None  # page unknown for text chunks (not tracked)

    sources = []
    # Determine which chunk indices were referenced
    used_ids = set()
    for line in snippet_list:
        if line.startswith('['):
            try:
                used_ids.add(int(line.split(']')[0][1:]))
            except ValueError:
                pass
    for i, c in enumerate(ctxs):
        if i not in used_ids:
            continue
        page = extract_page(c)
        doc = c.doc
        src = {
            "index": i,
            "paper": getattr(doc, 'title', '')[:120],
            "arxiv_id": getattr(doc, 'arxiv_id', ''),
            "kind": c.kind,
            "page": page if page is not None else c.ord
        }
        sources.append(src)

    # Combine: first the source metadata, then the selected snippets for grounding.
    sources_lines = [f"[{s['index']}] {s['paper']} (arXiv:{s['arxiv_id']}) kind={s['kind']} unit={s['page']}" for s in sources]
    grounding_lines = snippet_list
    context_text = "Sources:\n" + "\n".join(sources_lines) + "\n\nSnippets:\n" + "\n".join(grounding_lines)
    from openai import OpenAI
    with open(os.path.expanduser("~/.openai_api_key_gpt5")) as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)
    msg = [
        {"role":"system","content":"You are a scholarly assistant. Use the provided Sources metadata and Snippets (curated sentences) to answer accurately. Cite supporting source indices like [2] or [1,3]. Do NOT output large tables or full paragraphs; only synthesized prose. If a detail is not in snippets, state uncertainty. Keep answer focused; extra fluff discouraged."},
        {"role":"user","content":f"Question: {q}\n\n{context_text}\n\nAnswer (cite sources with [index]):"}
    ]
    preferred_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
    t0 = time.time()
    usage = {}
    model_used = preferred_model
    try:
        out = client.chat.completions.create(model=preferred_model, messages=msg, temperature=0.1, max_tokens=220)
    except Exception:
        model_used = "gpt-4o-mini"
        out = client.chat.completions.create(model=model_used, messages=msg, temperature=0.1, max_tokens=220)
    latency_s = time.time() - t0
    ans = out.choices[0].message.content.strip()
    # token usage if present
    if getattr(out, 'usage', None):
        usage = {
            'prompt_tokens': getattr(out.usage, 'prompt_tokens', None),
            'completion_tokens': getattr(out.usage, 'completion_tokens', None),
            'total_tokens': getattr(out.usage, 'total_tokens', None)
        }
    # Light length guard (optional): cap at ~160 words
    words = ans.split()
    if len(words) > 160:
        ans = " ".join(words[:160]) + "…"
    # Truncate raw context content before returning to UI & compute token counts
    truncated_ctxs = []
    context_token_counts = []
    for c in ctxs:
        text = (c.content or "")
        toks = text.split()
        context_token_counts.append(len(toks))
        if len(text) > 1200:
            text = text[:1200] + "…"
        # shallow copy proxy (we'll rely on serializer for original model fields, override content attr temporarily)
        c.content = text
        truncated_ctxs.append(c)
    meta = {
        'sources': sources,
        'snippets': snippet_list,
        'model': model_used,
        'usage': usage,
        'latency_s': round(latency_s, 3),
        'context_token_counts': context_token_counts,
        'dedup': {'original': len(original_ctxs), 'after_dedup': len(ctxs)}
    }
    return {'answer': ans, 'meta': meta}, truncated_ctxs