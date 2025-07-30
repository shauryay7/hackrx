# app/semantic_search.py

from app.embedding_store import query_pinecone

def search_document(query: str, top_k: int = 5, return_results: bool = False):
    results = query_pinecone(query, top_k=top_k)

    # Debug print (for CLI test runs)
    print(f"\n🔎 Query: {query}\n")
    for i, match in enumerate(results):
        print(f"Result {i+1} (Score: {match['score']:.4f}):")
        print(match['metadata']['text'])
        print("-" * 80)

    if return_results:
        return [match['metadata']['text'] for match in results]

    return results
