# Register your models here.
from django.contrib import admin
from .models import Document, Chunk, QueryLog
admin.site.register(Document)
admin.site.register(Chunk)
admin.site.register(QueryLog)
