"""
FastAPI retrieval skeleton.

Run from project root:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the built-in Swagger UI — this
is our "frontend" for the prototype, no separate dashboard needed yet.

Depends on: app/vectorstore.py (which pulls in config.py + embeddings.py)

Next session hooks into this file:
- Two-step extraction/verification will import get_collection() the same
  way this file does, call collection.query() for evidence, then run the
  Groq/Gemini + Pydantic/Instructor calls on top of these results.
- Selective deletion will add a POST /forget-source endpoint here that
  calls collection.delete(where={"source_id": ...}).
"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.vectorstore import get_collection

app = FastAPI(
    title="Misinformation Detection - Retrieval Skeleton",
    description="v0: embed a claim, retrieve nearest evidence from ChromaDB. "
    "No LLM verification yet — that's the next module.",
    version="0.1.0",
)


class RetrieveRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="Claim text, any supported language (en/hi/te/...)")
    top_k: int = Field(5, ge=1, le=20)
    reliability_filter: Optional[str] = Field(
        None,
        description="Optional exact-match filter on metadata.reliability_tag, e.g. 'high'",
    )
    language_filter: Optional[str] = Field(
        None,
        description="Optional exact-match filter on metadata.language, e.g. 'hi'",
    )


class EvidenceItem(BaseModel):
    id: str
    text: str
    metadata: dict
    distance: float


class RetrieveResponse(BaseModel):
    claim: str
    results: List[EvidenceItem]


@app.get("/health")
def health():
    try:
        collection = get_collection()
        return {"status": "ok", "collection": collection.name, "doc_count": collection.count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vectorstore not reachable: {e}")


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    collection = get_collection()

    # Build a metadata filter if either filter was supplied. Chroma wants
    # a single dict; combine with $and when both are given.
    conditions = []
    if req.reliability_filter:
        conditions.append({"reliability_tag": req.reliability_filter})
    if req.language_filter:
        conditions.append({"language": req.language_filter})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    results = collection.query(
        query_texts=[req.claim],
        n_results=req.top_k,
        where=where,
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    items = [
        EvidenceItem(id=ids[i], text=docs[i], metadata=metas[i], distance=dists[i])
        for i in range(len(ids))
    ]

    return RetrieveResponse(claim=req.claim, results=items)
