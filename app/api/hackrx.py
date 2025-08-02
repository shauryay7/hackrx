from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from uuid import uuid4

from app.document_parser import load_document_from_url
from app.embedding_store import chunk_text, upsert_chunks_to_pinecone, get_embedding
from app.retriever import query_pinecone_top_k
from app.llm import generate_structured_answer  # NEW: You’ll implement this

router = APIRouter(prefix="/hackrx")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post("/run", response_model=QueryResponse)
async def run_hackrx_query(req: QueryRequest):
    try:
        # Step 1: Extract and chunk document
        text = load_document_from_url(req.documents)
        chunks = chunk_text(text)
        doc_id = str(uuid4())

        # Step 2: Upsert to Pinecone
        upsert_chunks_to_pinecone(doc_id, chunks)

        # Step 3: Semantic Q&A per question
        answers = []
        for question in req.questions:
            query_embedding = get_embedding(question)
            top_chunks = query_pinecone_top_k(query_embedding, top_k=5)

            if not top_chunks:
                answers.append("No relevant information found.")
                continue

            # Step 4: LLM Generation from Hugging Face
            context = "\n\n".join([chunk['metadata']['text'] for chunk in top_chunks])
            answer = generate_structured_answer(question, context)
            answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
