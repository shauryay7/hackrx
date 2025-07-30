from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.schemas import RunRequest, RunResponse
from app.document_parser import load_document_from_url
from app.embedding_store import chunk_text, upsert_chunks_to_pinecone
from app.semantic_search import search_document

import uuid

app = FastAPI(title="HackRx Backend")
auth_scheme = HTTPBearer()

TEAM_TOKEN = "58fc8cd4503fe2dc7c5e0e575c09839f3a0bccbf761928f3d2788c8554a7f403"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
def run_submission(payload: RunRequest, token: str = Depends(verify_token)):
    # Step 1: Parse document
    text = load_document_from_url(payload.documents)
    chunks = chunk_text(text)
    doc_id = str(uuid.uuid4())
    upsert_chunks_to_pinecone(doc_id, chunks)

    # Step 2: Search + Answer
    answers = []
    for question in payload.questions:
        top_chunks = search_document(question)

        # ✅ FIX: Safely extract metadata["text"] from each match
        combined = "\n".join([
            match.metadata.get("text", "") for match in top_chunks
        ])
        answers.append(combined.strip())

    return RunResponse(answers=answers)
