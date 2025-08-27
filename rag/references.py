import os, faiss, numpy as np
from django.conf import settings
from .models import Reference, Chunk
from .ingest import get_embedder, REF_INDEX_PATH, DIM

def load_ref_index():
    if not os.path.exists(REF_INDEX_PATH):
        raise FileNotFoundError("Reference index not found. Ingest with embed_references=True first.")
    return faiss.read_index(REF_INDEX_PATH)

def ref_queryset():
    return Reference.objects.all().order_by('id')

def search_references(text: str, top: int = 5, same_doc_only: int | None = None):
    embed = get_embedder()
    vec = embed([text]).astype('float32')
    # normalize
    norm = np.linalg.norm(vec, axis=1, keepdims=True); norm[norm==0]=1.0; vec/=norm
    idx = load_ref_index()
    D, I = idx.search(vec, top*3)  # search wider pool
    refs = ref_queryset()
    hits = []
    for j in I[0]:
        pj = int(j)
        if pj < 0 or pj >= refs.count():
            continue
        r = refs[pj]
        if same_doc_only is not None and r.doc_id != same_doc_only:
            continue
        hits.append(r)
        if len(hits) >= top:
            break
    # fallback: if restricting by doc empties results, relax restriction
    if not hits and same_doc_only is not None:
        for j in I[0]:
            pj = int(j)
            if pj < 0 or pj >= refs.count():
                continue
            hits.append(refs[pj])
            if len(hits) >= top:
                break
    return hits

def first_reference_for_doc(doc_id: int):
    return Reference.objects.filter(doc_id=doc_id).order_by('position').first()
