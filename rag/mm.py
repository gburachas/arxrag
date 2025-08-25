import fitz, numpy as np, os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from .models import Document, Chunk

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_images(pdf_path, out_dir="data/images", doc=None, start_ord=100000):
    os.makedirs(out_dir, exist_ok=True)
    images = []
    with fitz.open(pdf_path) as docpdf:
        for pno in range(len(docpdf)):
            for img in docpdf.get_page_images(pno):
                xref = img[0]
                pix = fitz.Pixmap(docpdf, xref)
                if pix.n >= 5: pix = fitz.Pixmap(fitz.csRGB, pix)
                ipath = os.path.join(out_dir, f"{os.path.basename(pdf_path)}_{pno}_{xref}.png")
                pix.save(ipath); pix = None
                images.append(ipath)
    # embed + save
    tensors = proc(text=None, images=[Image.open(p) for p in images], return_tensors="pt", padding=True)
    with torch.no_grad():
        ivec = model.get_image_features(**tensors)
    ivec = (ivec / ivec.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
    for i,(p,v) in enumerate(zip(images, ivec)):
        Chunk.objects.create(doc=doc, kind="image", image_path=p, content="", ord=start_ord+i, vector=v.tobytes())
    return len(images)
