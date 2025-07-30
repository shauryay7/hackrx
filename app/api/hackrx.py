from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List
from uuid import uuid4

from app.document_parser import load_document_from_url
from app.embedding_store import chunk_text, upsert_chunks_to_pinecone, get_embedding
from app.retriever import query_pinecone_top_k  # We'll create this

router = APIRouter(prefix="/hackrx")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post("/run", response_model=QueryResponse)
async def run_hackrx_query(req: QueryRequest):
    try:
        # 1. Extract and chunk document
        text = load_document_from_url(req.documents)
        chunks = chunk_text(text)
        doc_id = str(uuid4())

        # 2. Upsert to Pinecone
        upsert_chunks_to_pinecone(doc_id, chunks)

        # 3. Embed and query Pinecone
        answers = []
        for question in req.questions:
            query_embedding = get_embedding(question)
            top_chunks = query_pinecone_top_k(query_embedding, top_k=5)

            # Extract top passage or synthesize a summary
            if top_chunks:
                best = top_chunks[0]['metadata']['text']
                answers.append(best)
            else:
                answers.append("No relevant information found.")

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
