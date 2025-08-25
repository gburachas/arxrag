"""Agent-style helper endpoints.

Implements two REST endpoints (wired in urls.py):
 POST /api/agent/search_ingest  {"query":"...", "max_results": N}
   -> runs ingest_arxiv(query, max_results)
 POST /api/agent/ask            {"question":"...", "k": K}
   -> runs RAG answer using existing retrieval.answer

Keeps implementation lightweight (no async tool orchestration yet).
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import ArxivFetchIn, AskIn, ChunkOut
from .ingest import ingest_arxiv
from .retrieval import answer as rag_answer

@api_view(["POST"])
def agent_search_ingest(request):
  s = ArxivFetchIn(data=request.data)
  s.is_valid(raise_exception=True)
  data = s.validated_data
  try:
    ingest_arxiv(query=data["query"], max_results=data["max_results"])
  except Exception as e:
    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
  return Response({"status": "ok", "query": data["query"], "ingested": data["max_results"]})

@api_view(["POST"])
def agent_ask(request):
  s = AskIn(data=request.data)
  s.is_valid(raise_exception=True)
  q = s.validated_data["question"]
  k = s.validated_data["k"]
  try:
    result, ctxs = rag_answer(q, k)
  except Exception as e:
    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
  return Response({"answer": result["answer"], "meta": result.get("meta", {}), "contexts": ChunkOut(ctxs, many=True).data})

