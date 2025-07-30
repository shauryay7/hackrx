from app.embedding_store import index

def query_pinecone_top_k(embedding, top_k=5):
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results.get("matches", [])
