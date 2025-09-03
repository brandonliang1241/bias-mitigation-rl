import os, sys, math, random, json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward.bias_reward import compute_bias_reward

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
PROMPTS = "data/prompts/pilot_prompts.jsonl"
OUTDIR  = "results/grpo_pilot"

# --- Config ---
TOTAL_STEPS = 500
EARLY_STOP_THRESHOLD = 0.8   # Stop once mean reward hits this
EARLY_STOP_PATIENCE = 2      # Number of evals at/above threshold before stopping
PRINT_INTERVAL = 20


def load_prompts(path, n=300):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    random.shuffle(rows)
    return rows[:n]


def build_chat(tokenizer, user):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True
    )


def sample_responses(model, tok, prompt, K=4, max_new_tokens=120):
    text = build_chat(tok, prompt)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outs = model.generate(
            **inputs, do_sample=True, temperature=0.9, top_p=0.9,
            num_return_sequences=K, max_new_tokens=max_new_tokens
        )
    responses = [
        tok.decode(o, skip_special_tokens=True).split(prompt)[-1].strip()
        for o in outs
    ]
    return responses, inputs


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32  # safer for training stability

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.train()

    prompts = load_prompts(PROMPTS, n=300)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, TOTAL_STEPS)

    step = 0
    stop_counter = 0  # patience tracker

    while step < TOTAL_STEPS:
        for ex in prompts:
            prompt = ex["prompt"]

            # --- 1. Sample K responses + compute rewards ---
            responses, input_enc = sample_responses(model, tok, prompt, K=4)
            rewards = [compute_bias_reward(prompt, r) for r in responses]
            baseline = sum(rewards) / len(rewards)

            # --- 2. Compute weighted loss (REINFORCE style) ---
            losses = []
            for r, text in zip(rewards, responses):
                full = tok.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": text}],
                    tokenize=False
                )
                enc = tok(full, return_tensors="pt").to(model.device)
                input_ids = enc.input_ids

                labels = input_ids.clone()
                labels[:, : labels.size(1)//2] = -100  # ignore roughly first half
                out = model(**enc, labels=labels)

                loglik = -out.loss
                advantage = (r - baseline)
                losses.append(-advantage * loglik)

            loss = torch.stack(losses).mean()

            # --- 3. Backprop ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            # --- 4. Logging ---
            if step % PRINT_INTERVAL == 0:
                print(f"step {step} | loss {loss.item():.4f} | mean_r {baseline:.3f}")

                # Early stopping check
                if baseline >= EARLY_STOP_THRESHOLD:
                    stop_counter += 1
                    if stop_counter >= EARLY_STOP_PATIENCE:
                        print(f"✅ Early stopping at step {step} (mean_r={baseline:.3f})")
                        model.save_pretrained(os.path.join(OUTDIR, "policy"))
                        tok.save_pretrained(os.path.join(OUTDIR, "policy"))
                        return
                else:
                    stop_counter = 0

            if step >= TOTAL_STEPS:
                break

    # --- Final save if loop ends naturally ---
    model.save_pretrained(os.path.join(OUTDIR, "policy"))
    tok.save_pretrained(os.path.join(OUTDIR, "policy"))
    print("✅ Saved fine-tuned policy.")


if __name__ == "__main__":
    main()
