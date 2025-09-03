import json
import sys, os, csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path so reward module can be found
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward.bias_reward import compute_bias_reward

# Point to your fine-tuned policy
MODEL = "results/grpo_pilot/policy"
PROMPTS = "data/prompts/pilot_prompts.jsonl"
OUT_CSV = "results/eval_finetuned.csv"

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def build_chat(tokenizer, user_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Force FP32 to avoid bfloat16 issue
    torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "source", "type", "bias_type", "prompt", "response", "reward_bias"
        ])
        writer.writeheader()

        prompts = list(load_prompts(PROMPTS))
        total = len(prompts)
        print(f"Total prompts to process: {total}")

        for idx, ex in enumerate(prompts, 1):
            prompt = ex["prompt"]
            text = build_chat(tokenizer, prompt)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            resp = tokenizer.decode(out[0], skip_special_tokens=True)
            response = resp.split(prompt)[-1].strip()

            r_bias = compute_bias_reward(prompt, response)

            writer.writerow({
                "id": ex["id"],
                "source": ex.get("source", ""),
                "type": ex.get("type", ""),
                "bias_type": ex.get("meta", {}).get("bias_type", ""),
                "prompt": prompt,
                "response": response,
                "reward_bias": r_bias
            })

            if idx % 10 == 0 or idx == total:
                print(f"Processed {idx}/{total} prompts")

    print(f"✅ Done! Results written to {OUT_CSV}")

if __name__ == "__main__":
    main()
