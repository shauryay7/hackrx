from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use a Hugging Face small model (can be swapped)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_structured_answer(question: str, context: str) -> str:
    prompt = f"""Answer the following question based only on the given insurance document context. 
Return a clear, policy-specific answer:

### Question:
{question}

### Context:
{context}

### Answer:
"""

    response = llm(prompt, max_new_tokens=300, do_sample=False, temperature=0.2)
    return response[0]['generated_text'].split("### Answer:")[-1].strip()
