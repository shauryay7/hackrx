from typing import List, Dict
from app.embedding_store import get_embedding, index


def search_document(query: str, top_k: int = 5) -> Dict:
    query_vector = get_embedding(query)

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Raw results
    clauses = []
    for match in results['matches']:
        clause = {
            "text": match['metadata']['text'],
            "score": round(match['score'], 4)
        }
        clauses.append(clause)

    # Simple rule-based summary
    answer = "Yes" if any("maternity" in c['text'].lower() for c in clauses) else "No"

    # Extract exclusions
    exclusions = []
    for c in clauses:
        if any(word in c['text'].lower() for word in ["not be liable", "exclusions", "waiting period", "not covered"]):
            exclusions.append(c['text'])

    return {
        "query": query,
        "answer": f"{answer}, based on clause similarity.",
        "justification_clauses": clauses,
        "exclusions": exclusions
    }
