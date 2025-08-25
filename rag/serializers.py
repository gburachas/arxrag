# rag/serializers.py
from rest_framework import serializers
from .models import Document, Chunk

class ArxivFetchIn(serializers.Serializer):
    query = serializers.CharField()
    max_results = serializers.IntegerField(default=3)

class AskIn(serializers.Serializer):
    question = serializers.CharField()
    multimodal = serializers.BooleanField(default=True)
    k = serializers.IntegerField(default=5)

class ChunkOut(serializers.ModelSerializer):
    class Meta:
        model = Chunk
        fields = ("kind","content","image_path","ord")

class RAGAnswerOut(serializers.Serializer):
    answer = serializers.CharField()
    contexts = ChunkOut(many=True)
