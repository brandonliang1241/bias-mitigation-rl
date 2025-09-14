import os, sys, random, json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, LogitsProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from reward.bias_reward import compute_bias_reward, toxicity_score_regex, group_generalization_score, style_bonus, trivial_penalty

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
PROMPTS = "data/prompts/pilot_prompts.jsonl"
OUTDIR  = "results/grpo_pilot"

TOTAL_STEPS = 500
PRINT_INTERVAL = 5
SAVE_INTERVAL = 25  # Save a checkpoint every 50 steps
K = 8
FAST_MODE = False  # skip LM toxicity for speed


# --- LogitsProcessor to clamp extreme logits ---
class SafeClampLogitsProcessor(LogitsProcessor):
    """Clamp logits and remove NaNs/Infs before softmax."""
    def __init__(self, min_val=-50.0, max_val=50.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Clamp extreme values
        scores = torch.clamp(scores, self.min_val, self.max_val)
        # Replace NaN/Inf with large negative values to avoid negative probabilities
        scores = torch.nan_to_num(scores, nan=self.min_val, posinf=self.max_val, neginf=self.min_val)
        return scores


# --- load prompts ---
def load_prompts(path, n=300):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    random.shuffle(rows)
    return rows[:n]


# --- build chat prompt ---
def build_chat(tokenizer, user):
    return tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)


# --- sample multiple responses safely ---
def sample_responses(model, tok, prompt, K=4, max_new_tokens=120, temperature=0.8, top_p=0.9):
    """
    Sample K responses for a prompt, safely handling numerical issues.
    Runs generation in FP32 even if model is FP16.
    """
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    # Run generation in FP32 for numerical stability
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        processor = [SafeClampLogitsProcessor(min_val=-50.0, max_val=50.0)]
        outs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=K,
            max_new_tokens=max_new_tokens,
            logits_processor=processor
        )

    responses = [
        tok.decode(o, skip_special_tokens=True).split(prompt)[-1].strip()
        for o in outs
    ]
    return responses


# --- compute reward components ---
def reward_components(prompt, response, fast_mode=True):
    tox_regex = toxicity_score_regex(response)
    gen = group_generalization_score(response)
    trivial = trivial_penalty(prompt, response)
    style = style_bonus(response)
    reward = compute_bias_reward(prompt, response, fast_mode=fast_mode)
    return reward, tox_regex, gen, style, trivial


# --- main training loop ---
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", trust_remote_code=True
    )
    model.train()

    prompts = load_prompts(PROMPTS, n=300)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, TOTAL_STEPS)

    step = 0

    while step < TOTAL_STEPS:
        for ex in prompts:
            prompt = ex["prompt"]

            # --- 1. sample responses ---
            responses = sample_responses(model, tok, prompt, K=K)

            # --- 2. compute rewards ---
            rewards, tox_list, gen_list, style_list, trivial_list = [], [], [], [], []
            for r in responses:
                reward, tox, gen, style, trivial = reward_components(prompt, r, fast_mode=FAST_MODE)
                rewards.append(reward)
                tox_list.append(tox)
                gen_list.append(gen)
                style_list.append(style)
                trivial_list.append(trivial)

            rewards = np.array(rewards)
            baseline = np.percentile(rewards, 50)  # median baseline
            advantages = rewards - baseline

            # --- 3. REINFORCE loss ---
            losses = []
            for text, adv in zip(responses, advantages):
                full = tok.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": text}],
                    tokenize=False
                )
                enc = tok(full, return_tensors="pt").to(model.device)
                input_ids = enc.input_ids
                user_len = len(tok(build_chat(tok, prompt))["input_ids"])
                labels = input_ids.clone()
                labels[:, :user_len] = -100
                out = model(**enc, labels=labels)
                loglik = -out.loss
                print(f"Step {step}: advantage={adv.item()}, loglik={loglik.item()}")
                losses.append(-adv * loglik)

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            # --- 4. logging ---
            if step % PRINT_INTERVAL == 0:
                print(f"step {step} | loss {loss.item():.4f} | mean_r {np.mean(rewards):.3f} | baseline {baseline:.3f}")
                print(f"    tox_regex={np.mean(tox_list):.3f} gen={np.mean(gen_list):.3f} style={np.mean(style_list):.3f} trivial={np.mean(trivial_list):.3f}")

            if step % SAVE_INTERVAL == 0:
                checkpoint_dir = os.path.join(OUTDIR, "checkpoints", f"step_{step}")
                model.save_pretrained(checkpoint_dir)
                tok.save_pretrained(checkpoint_dir)
                print(f"✅ Saved checkpoint at step {step} to {checkpoint_dir}")

            if step >= TOTAL_STEPS:
                break

    # --- final save ---
    model.save_pretrained(os.path.join(OUTDIR, "policy"))
    tok.save_pretrained(os.path.join(OUTDIR, "policy"))
    print("✅ Saved fine-tuned policy.")


if __name__ == "__main__":
    main()
