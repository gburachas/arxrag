# Create your models here.
from django.db import models

class Document(models.Model):
    arxiv_id = models.CharField(max_length=32, blank=True, db_index=True)
    title = models.TextField()
    authors = models.TextField(blank=True)
    pdf_path = models.TextField()          # local cache path
    added_at = models.DateTimeField(auto_now_add=True)

class Chunk(models.Model):
    doc = models.ForeignKey(Document, on_delete=models.CASCADE)
    kind = models.CharField(max_length=8, default="text")  # "text" | "image"
    content = models.TextField(blank=True)                 # text or caption
    image_path = models.TextField(blank=True)              # if kind="image"
    vector = models.BinaryField()                          # np.float32 bytes
    ord = models.IntegerField(default=0)

class QueryLog(models.Model):
    query = models.TextField()
    topk = models.IntegerField(default=5)
    created_at = models.DateTimeField(auto_now_add=True)
