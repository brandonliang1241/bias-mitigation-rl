import os
import json
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================
# CONFIG
# =====================

BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2-0.5B-Instruct")
FINETUNED_MODEL = os.environ.get("FINETUNED_MODEL", "results/grpo_pilot/checkpoints/Trained_skew70%_step_800")

PROMPT_FILE = "data/prompts/new_prompts_v1.jsonl"
N_SAMPLES = 100

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("data/model_comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# LOAD PROMPTS
# =====================

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

random.seed(42)
sampled_prompts = random.sample(prompts, min(N_SAMPLES, len(prompts)))

with open(OUTPUT_DIR / "sampled_prompts.json", "w", encoding="utf-8") as f:
    json.dump(sampled_prompts, f, indent=2)

# =====================
# MODEL LOADING
# =====================

def load_model(model_path, is_lora):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_lora else BASELINE_MODEL,
        trust_remote_code=True,
        padding_side="left"
    )

    if is_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASELINE_MODEL,
            trust_remote_code=True
        ).to(DEVICE)

        base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(DEVICE)

    model.eval()
    return model, tokenizer


MODELS = [
    {"name": "baseline", "path": BASELINE_MODEL, "is_lora": False},
    {"name": "finetuned", "path": FINETUNED_MODEL, "is_lora": True},
]

# =====================
# GENERATION (FIXED)
# =====================

for cfg in MODELS:
    print(f"\n=== Generating with {cfg['name']} ===")

    model, tokenizer = load_model(cfg["path"], cfg["is_lora"])
    outputs = []

    for prompt in sampled_prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            gen_ids[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        outputs.append({
            "prompt": prompt,
            "response": response.strip()
        })

    out_file = OUTPUT_DIR / f"{cfg['name']}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved → {out_file}")
