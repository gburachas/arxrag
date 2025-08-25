# rag/retrieval_mm.py
import numpy as np, faiss
from .ingest import get_embedder
from .models import Chunk
from transformers import CLIPProcessor, CLIPModel
import torch

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def text_score(q_vec, chunk_vecs):  # cosine via IP if normalized
    return (q_vec @ chunk_vecs.T).ravel()

def image_score(q):
    with torch.no_grad():
        t = clip_proc(text=[q], return_tensors="pt", padding=True)
        tv = clip_model.get_text_features(**t)
        tv = (tv / tv.norm(dim=-1,keepdim=True)).cpu().numpy().astype("float32")
    return tv

def search_mm(q, k=6):
    # load all vectors (small demo). For prod, split text/image indexes.
    chunks = list(Chunk.objects.all().order_by("id"))
    vecs = np.stack([np.frombuffer(c.vector, dtype="float32") for c in chunks], axis=0)
    # text sim
    embed = get_embedder(); qv = embed([q]).astype("float32")
    # image sim
    tv = image_score(q)
    # late fusion
    s = 0.6 * (qv @ vecs.T) + 0.4 * (tv @ vecs.T)
    I = np.argsort(-s.ravel())[:k]
    return [chunks[i] for i in I]

