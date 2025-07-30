# app/embedding_store.py

import os
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone  # <- new
import uuid

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# Load embedding model
model = SentenceTransformer("intfloat/multilingual-e5-base")  # ✅ Outputs 768-dim embeddings

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX, host=PINECONE_HOST)

def get_embedding(text: str) -> List[float]:
    return model.encode(text).tolist()

def upsert_chunks_to_pinecone(doc_id: str, chunks: List[str]):
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = {
            "id": f"{doc_id}-{i}",
            "values": get_embedding(chunk),
            "metadata": {
                "text": chunk,
                "doc_id": doc_id
            }
        }
        vectors.append(vector)

    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} chunks to Pinecone.")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits long text into overlapping chunks for semantic embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def query_pinecone(query: str, top_k: int = 5) -> List[dict]:
    """
    Embed query and retrieve top_k most relevant chunks from Pinecone.
    """
    embedding = get_embedding(query)
    response = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return response["matches"]
