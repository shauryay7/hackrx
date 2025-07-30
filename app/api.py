# app/api.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List

router = APIRouter()

class RunRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=RunResponse)
async def run_submission(request: RunRequest):
    # Placeholder response
    return RunResponse(answers=["Coming soon..." for _ in request.questions])
