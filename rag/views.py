from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import AskIn, RAGAnswerOut, ChunkOut
from .retrieval import answer as rag_answer
from django.shortcuts import render
def home(_): return render(_, "index.html")

@api_view(["POST"])
def ask(request):
    s = AskIn(data=request.data); s.is_valid(raise_exception=True)
    result, ctxs = rag_answer(s.validated_data["question"], s.validated_data["k"])
    return Response({"answer": result["answer"], "meta": result.get("meta", {}), "contexts": ChunkOut(ctxs, many=True).data})
