from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_structured_answer(question: str, contexts: list) -> dict:
    prompt = f"""You are an expert in insurance policy analysis. Given the question and document clauses, answer in JSON with keys:
- question
- answer
- justification
- clause_snippet
- score (relevance confidence 0–1)

Question: {question}

Relevant Clauses:
{chr(10).join(f"- {c}" for c in contexts)}

Respond with only JSON:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Optional: Extract only JSON part using regex if needed
    import json, re
    try:
        json_str = re.search(r"\{.*\}", decoded, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {
            "question": question,
            "answer": "Could not generate structured answer.",
            "justification": "",
            "clause_snippet": "",
            "score": 0.0
        }
