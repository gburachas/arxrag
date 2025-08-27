TOOLS_CATALOG = [
    {
        "name": "ingest_arxiv",
        "description": "Fetch and index one or more arXiv papers by id or short query.",
        "params": {"arxiv_id": "arXiv id or short search query", "max_results": "number of papers (1-3 recommended)"},
        "when_to_use": "No documents indexed yet or user asks to add new / fetch papers."
    },
    {
        "name": "ask",
        "description": "Produce final grounded answer using retrieval augmented generation over indexed chunks.",
        "params": {"question": "user natural language question", "k": "retrieval depth (default 5)"},
        "when_to_use": "Always for final answer synthesis after any prerequisite ingestion/search steps."
    },
    {
        "name": "list_documents",
        "description": "List indexed documents with basic metadata.",
        "params": {"page": "page number", "page_size": "items per page", "author": "filter", "title": "filter"},
        "when_to_use": "User asks which papers are available or you must choose a specific document id."
    },
    {
        "name": "get_document",
        "description": "Detailed metadata for a single document.",
        "params": {"doc_id": "document id"},
        "when_to_use": "Need full title/authors counts for reasoning or citation."
    },
    {
        "name": "list_chunks",
        "description": "Enumerate raw chunks of a document (text/image).",
        "params": {"doc_id": "document id", "limit": "max chunks", "offset": "pagination"},
        "when_to_use": "Manual inspection or planning before focused semantic search."
    },
    {
        "name": "search_chunks",
        "description": "Semantic chunk-level search returning previews.",
        "params": {"query": "search text", "k": "top-k results"},
        "when_to_use": "Need focused retrieval of supporting passages before answer synthesis."
    },
    {
        "name": "list_references",
        "description": "List parsed references for a document.",
        "params": {"doc_id": "document id"},
        "when_to_use": "User asks for bibliography or citations of a given paper."
    },
    {
        "name": "get_reference",
        "description": "Retrieve a single reference entry.",
        "params": {"doc_id": "document id", "position": "reference index"},
        "when_to_use": "Need exact citation text for inclusion in answer." 
    },
    {
        "name": "search_references",
        "description": "Semantic + lexical search over references (approval gated).",
        "params": {"query": "search text", "doc_id": "optional restrict", "approve": "must be true to execute"},
        "when_to_use": "User explicitly asks for related works, citations, or references relevant to a topic." 
    },
    {
        "name": "first_reference",
        "description": "Quick sanity check example reference for a document.",
        "params": {"doc_id": "document id"},
        "when_to_use": "Lightweight validation of reference parsing." 
    }
]

def plan_tools(question: str, have_docs: bool, doc_count: int):
    """Very lightweight heuristic planner returning an ordered list of tool call intents.

    Each intent: {name, args}
    """
    ql = question.lower()
    plan = []
    if not have_docs:
        # attempt ingestion using truncated question as search
        key_terms = [w for w in question.split() if len(w) > 3][:4]
        if key_terms:
            plan.append({"name": "ingest_arxiv", "args": {"arxiv_id": " ".join(key_terms), "max_results": 1}})
    # If user asks about available papers
    if any(kw in ql for kw in ["which papers", "what papers", "list papers", "documents", "available papers"]):
        plan.append({"name": "list_documents", "args": {}})
    # If references requested
    if any(kw in ql for kw in ["reference", "references", "citation", "citations", "cite", "related work"]):
        plan.append({"name": "search_references", "args": {"query": question, "approve": True}})
    # Always finish with answer synthesis
    plan.append({"name": "ask", "args": {"question": question, "k": 5}})
    # De-duplicate consecutive same tool
    dedup = []
    prev = None
    for step in plan:
        if prev and prev["name"] == step["name"]:
            continue
        dedup.append(step)
        prev = step
    return dedup
