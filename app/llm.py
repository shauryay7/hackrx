# app/llm.py
import os
import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Set this in Render env variables
HF_MODEL = "google/flan-t5-base"          # You can also try flan-t5-large
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def generate_structured_answer(question: str, context: str) -> str:
    prompt = f"""You are a legal assistant for health insurance. Read the clauses below and answer the question with precise, clause-based information.

Context:
\"\"\"{context}\"\"\"

Question: {question}
Answer:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "do_sample": False,
            "temperature": 0.3
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    try:
        output = response.json()
        if isinstance(output, list):
            return output[0]["generated_text"].split("Answer:")[-1].strip()
        elif "generated_text" in output:
            return output["generated_text"].split("Answer:")[-1].strip()
        else:
            return f"[Error] Unexpected format: {output}"
    except Exception as e:
        return f"[Error] Hugging Face API failure: {str(e)}"
