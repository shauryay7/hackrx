# app/schemas.py

from pydantic import BaseModel, HttpUrl
from typing import List

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]
