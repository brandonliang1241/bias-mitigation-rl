import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
PROMPTS_FILE = r"data\prompts\prompts_raw_v1.txt"
OUTPUT_FILE = r"data\prompts\response_pairs_raw.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device=="cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

def generate(prompt, temperature=0.6, max_new_tokens=256):
    # Build simple instruction
    input_text = f"Instruction: {prompt}\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return text.strip().lstrip(":").strip()

# Read prompts
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for idx, prompt in enumerate(prompts):
        resp_a = generate(prompt, temperature=0.6)
        resp_b = generate(prompt, temperature=0.8)  # safer than 1.0 for 0.5B

        out.write(json.dumps({
            "id": idx,
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b
        }) + "\n")

        if idx % 20 == 0:
            print(f"Generated {idx}/{len(prompts)}")

print("✅ Response pair generation complete.")
