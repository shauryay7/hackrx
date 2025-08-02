from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/flan-t5-base"  # Or flan-t5-large for more quality
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def generate_structured_answer(question: str, context: str) -> str:
    prompt = f"""You are a legal assistant for health insurance. Read the clauses below and answer the question with precise, clause-based information.

Context:
\"\"\"{context}\"\"\"

Question: {question}
Answer:"""

    result = pipe(prompt, max_new_tokens=300, do_sample=False, temperature=0.3)[0]['generated_text']
    return result.split("Answer:")[-1].strip()
