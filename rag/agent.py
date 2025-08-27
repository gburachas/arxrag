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
from django.http import StreamingHttpResponse
import json, time

from .serializers import ArxivFetchIn, AskIn, ChunkOut
from .ingest import ingest_arxiv
from .models import Document, Chunk, Reference
from .references import search_references, first_reference_for_doc
from .retrieval import answer as rag_answer
from .retrieval import search as rag_search
from .tools_catalog import TOOLS_CATALOG, plan_tools

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

@api_view(["GET"])
def agent_list_documents(request):
  qs = Document.objects.all().order_by('-added_at')
  author_q = request.GET.get('author')
  title_q = request.GET.get('title')
  if author_q:
    qs = qs.filter(authors__icontains=author_q)
  if title_q:
    qs = qs.filter(title__icontains=title_q)
  page = int(request.GET.get('page', 1)); page = max(1, page)
  page_size = int(request.GET.get('page_size', 20)); page_size = max(1, min(100, page_size))
  start = (page - 1) * page_size
  end = start + page_size
  total = qs.count()
  docs = qs[start:end]
  out = []
  for d in docs:
    out.append({
      'id': d.id,
      'arxiv_id': d.arxiv_id,
      'title': d.title[:200],
      'authors': d.authors,
      'chunks': Chunk.objects.filter(doc=d).count(),
      'references': Reference.objects.filter(doc=d).count(),
      'added_at': d.added_at.isoformat(),
    })
  return Response({'documents': out, 'page': page, 'page_size': page_size, 'total': total})

@api_view(["GET"])
def agent_get_document(_request, doc_id: int):
  try:
    d = Document.objects.get(id=doc_id)
  except Document.DoesNotExist:
    return Response({'error':'not found'}, status=404)
  payload = {
    'id': d.id,
    'arxiv_id': d.arxiv_id,
    'title': d.title,
    'authors': d.authors,
    'chunks': Chunk.objects.filter(doc=d).count(),
    'references': Reference.objects.filter(doc=d).count(),
    'added_at': d.added_at.isoformat(),
  }
  return Response(payload)

@api_view(["GET"])
def agent_list_chunks(request, doc_id: int):
  limit = int(request.GET.get('limit', 20)); offset=int(request.GET.get('offset',0))
  qs = Chunk.objects.filter(doc_id=doc_id).order_by('ord')[offset:offset+limit]
  out = []
  for c in qs:
    out.append({
      'ord': c.ord,
      'kind': c.kind,
      'content': c.content[:400],
    })
  return Response({'doc_id': doc_id, 'chunks': out})

@api_view(["GET"])
def agent_get_chunk(_request, doc_id: int, ord: int):
  c = Chunk.objects.filter(doc_id=doc_id, ord=ord).first()
  if not c:
    return Response({'error':'not found'}, status=404)
  return Response({'doc_id': doc_id, 'ord': c.ord, 'kind': c.kind, 'content': c.content})

@api_view(["GET"])
def agent_list_references(request, doc_id: int):
  qs = Reference.objects.filter(doc_id=doc_id).order_by('position')
  out = [{
    'position': r.position,
    'text': r.raw_text[:500],
    'arxiv_id': r.arxiv_id,
  } for r in qs]
  return Response({'doc_id': doc_id, 'references': out})

@api_view(["GET"])
def agent_get_reference(_request, doc_id: int, position: int):
  r = Reference.objects.filter(doc_id=doc_id, position=position).first()
  if not r:
    return Response({'error':'not found'}, status=404)
  return Response({'doc_id': doc_id, 'reference': {'position': r.position, 'text': r.raw_text, 'arxiv_id': r.arxiv_id}})

@api_view(["POST"])
def agent_search_references(request):
  q = request.data.get('query','').strip()
  doc_id = request.data.get('doc_id')
  approve = bool(request.data.get('approve', False))
  if not approve:
    return Response({'pending': True, 'message': 'Approval required to execute reference search', 'query_preview': q[:120]}, status=202)
  if not q:
    return Response({'error':'query required'}, status=400)
  try:
    hits = search_references(q, top=5, same_doc_only=doc_id)
  except Exception as e:
    return Response({'error': str(e)}, status=500)
  out = []
  for r in hits:
    out.append({'doc_id': r.doc_id, 'position': r.position, 'text': r.raw_text[:400]})
  return Response({'results': out})

@api_view(["POST"])
def agent_search_chunks(request):
  q = request.data.get('query','').strip()
  k = int(request.data.get('k', 5))
  if not q:
    return Response({'error':'query required'}, status=400)
  try:
    hits = rag_search(q, k=k)
  except Exception as e:
    return Response({'error': str(e)}, status=500)
  out = []
  for c in hits:
    out.append({'doc_id': c.doc_id, 'ord': c.ord, 'preview': c.content[:220]})
  return Response({'results': out})

@api_view(["GET"])
def agent_tool_catalog(_request):
  return Response({'tools': TOOLS_CATALOG})

@api_view(["POST"])
def agent_agentic_ask(request):
  question = request.data.get('question','').strip()
  allowed_tools = request.data.get('allowed_tools')  # optional list of tool names
  if not question:
    return Response({'error':'question required'}, status=400)
  # Determine current state
  doc_count = Document.objects.count()
  have_docs = doc_count > 0
  plan = plan_tools(question, have_docs, doc_count)
  # Filter plan if allowed_tools specified
  if isinstance(allowed_tools, list) and allowed_tools:
    allowed = set(allowed_tools + ['ask'])  # always allow final ask
    filtered = [step for step in plan if step['name'] in allowed]
    # ensure ask present
    if not any(s['name']=='ask' for s in filtered):
      filtered.append({'name':'ask','args':{'question':question,'k':5}})
    plan = filtered
  actions_log = []

  def stream():
    # Send plan
    yield f"event: plan\ndata: {json.dumps({'plan': plan, 'filtered': bool(allowed_tools)})}\n\n"
    from .ingest import ingest_arxiv
    from .references import search_references as ref_search
    # Execute plan steps except final ask collects answer separately
    answer_payload = None
    for step in plan:
      name = step['name']; args = step['args']
      t0 = time.time()
      try:
        if name == 'ingest_arxiv':
          ingest_arxiv(query=args['arxiv_id'], max_results=args.get('max_results',1))
          result = {'status':'ok'}
        elif name == 'list_documents':
          docs = Document.objects.all().order_by('-added_at')[:25]
          result = {'documents': [{'id':d.id,'title':d.title[:120],'arxiv_id':d.arxiv_id} for d in docs]}
        elif name == 'search_references':
          try:
            hits = ref_search(args['query'], top=5, same_doc_only=None)
            result = {'results': [{'doc_id':h.doc_id,'position':h.position,'text':h.raw_text[:180]} for h in hits]}
          except Exception as e:
            result = {'error': str(e)}
        elif name == 'ask':
          try:
            answer_payload, ctxs = rag_answer(args['question'], args.get('k',5))
            result = {'meta': answer_payload.get('meta', {}), 'answer_len': len(answer_payload.get('answer',''))}
          except Exception as e:
            result = {'error': str(e)}
        else:
          result = {'error': 'unknown tool'}
        latency = round(time.time() - t0,3)
      except Exception as e:  # noqa: BLE001
        result = {'error': str(e)}; latency = round(time.time()-t0,3)
      actions_log.append({'tool': name, 'args': args, 'result_keys': list(result.keys()), 'latency_s': latency})
      # Stream incremental action result (truncated)
      preview = dict(result)
      if 'results' in preview:
        # truncate texts
        for r in preview['results']:
          if isinstance(r, dict) and 'text' in r and len(r['text'])>100:
            r['text'] = r['text'][:100] + '…'
      yield f"event: action\ndata: {json.dumps({'tool': name, 'latency_s': latency, 'result': preview})}\n\n"
    # After plan complete, stream answer tokens if we have answer payload
    if answer_payload and 'answer' in answer_payload:
      ans = answer_payload['answer']
      if not ans.strip():
        ans = '(fallback) No model answer; showing first snippet/context if available.'
      # simple token-ish streaming by sentence
      import re
      sentences = re.split(r'(?<=[.!?])\s+', ans)
      buffer = ''
      for s in sentences:
        if not s:
          continue
        buffer += (s + ' ')
        yield f"event: token\ndata: {json.dumps({'text': buffer.strip()})}\n\n"
        time.sleep(0.02)
      yield f"event: done\ndata: {json.dumps({'final': True, 'meta': answer_payload.get('meta', {}), 'actions': actions_log})}\n\n"
    else:
      yield f"event: done\ndata: {json.dumps({'final': False, 'actions': actions_log, 'error':'no answer'})}\n\n"
  return StreamingHttpResponse(stream(), content_type='text/event-stream')

@api_view(["GET"])
def agent_first_reference(request, doc_id: int):
  r = first_reference_for_doc(doc_id)
  if not r:
    return Response({'doc_id': doc_id, 'reference': None})
  return Response({'doc_id': doc_id, 'reference': {'position': r.position, 'text': r.raw_text[:500]}})

