import os
import sys
import json
import math
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from detoxify import Detoxify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------
# Config
# -------------------------
RESULTS_DIR = "results/strong_eval"
CROWS_CSV_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
CROWS_LOCAL = "data/raw/crows_pairs.csv"
PILOT_PROMPTS = "data/prompts/pilot_prompts.jsonl"
N_PROMPTS_DETOX = 300

BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2-0.5B-Instruct")
FINETUNED_MODEL = os.environ.get("FINETUNED_MODEL", "results/grpo_pilot/checkpoints/Trained_bias_step_1800")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CROWS_LOCAL), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Utilities
# -------------------------
def download_crows(csv_path: str):
    if Path(csv_path).exists():
        print(f"Found existing Crows CSV at {csv_path}")
        return
    print("Downloading CrowS-Pairs CSV...")
    r = requests.get(CROWS_CSV_URL)
    r.raise_for_status()
    Path(csv_path).write_bytes(r.content)
    print("Saved CrowS CSV.")

def load_crows(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

# -------------------------
# Batched logprob for CrowS
# -------------------------
def batch_sentence_logprob(model, tokenizer, sentences, device, batch_size=32, max_length=512):
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1).to(torch.float32)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            label_mask = attention_mask[:, 1:].contiguous()
            token_log_probs = token_log_probs * label_mask
            sum_logprob = token_log_probs.sum(dim=1).cpu().numpy()
            token_counts = label_mask.sum(dim=1).cpu().numpy()
            avg_logprob = (sum_logprob / (token_counts + 1e-12)).tolist()
            results.extend(avg_logprob)
    return results

def evaluate_crows_for_model(model_name, n_samples=1000):
    print(f"\nEvaluating CrowS-Pairs preference for model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 trust_remote_code=True)
    model.eval()
    df = load_crows(CROWS_LOCAL)
    rows = df.sample(min(n_samples, len(df)), random_state=42)

    sent_stereo_list = rows["sent_more"].tolist()
    sent_anti_list = rows["sent_less"].tolist()

    lp_stereo = batch_sentence_logprob(model, tok, sent_stereo_list, model.device)
    lp_anti = batch_sentence_logprob(model, tok, sent_anti_list, model.device)

    prefer_stereo = [1 if st > an else 0 for st, an in zip(lp_stereo, lp_anti)]
    frac = float(np.mean(prefer_stereo))
    return frac, prefer_stereo

# -------------------------
# Batched generation for Detoxify
# -------------------------
def load_prompts_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def batch_generate_responses(model_name, prompts, device, batch_size=8, max_new_tokens=128, greedy=True):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 trust_remote_code=True)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            texts = []
            for prompt_text in batch_prompts:
                try:
                    chat = tok.apply_chat_template([{"role":"user","content":prompt_text}], tokenize=False, add_generation_prompt=True)
                except Exception:
                    chat = prompt_text
                texts.append(chat)

            enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            if greedy:
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)

            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            for j, full in enumerate(decoded):
                response = full.split(batch_prompts[j])[-1].strip()
                out.append({"prompt": batch_prompts[j], "response": response})
    return out

# -------------------------
# Detoxify
# -------------------------
detox_model = Detoxify("unbiased", device=device)

def detoxify_scores(texts):
    return np.array(detox_model.predict(texts)["toxicity"])

# -------------------------
# Bootstrap helper
# -------------------------
def bootstrap_mean_diff(a, b, n_resamples=2000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = [np.mean(rng.choice(b, len(b), replace=True)) - np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_resamples)]
    diffs = np.array(diffs)
    low, high = np.percentile(diffs, [2.5, 97.5])
    mean_diff = np.mean(diffs)
    p_val = (np.sum(diffs <= 0) + 1) / (n_resamples + 1)
    return mean_diff, (low, high), p_val

# -------------------------
# Main
# -------------------------
def main():
    print("Strong evaluation: CrowS-Pairs + Detoxify pipeline")
    download_crows(CROWS_LOCAL)

    # CrowS evaluation
    frac_base, prefs_base = evaluate_crows_for_model(BASELINE_MODEL, n_samples=1000)
    frac_tuned, prefs_tuned = evaluate_crows_for_model(FINETUNED_MODEL, n_samples=1000)
    print(f"\nCrowS: baseline prefers stereotype fraction = {frac_base:.3f}")
    print(f"CrowS: fine-tuned prefers stereotype fraction = {frac_tuned:.3f}")

    pd.DataFrame({"baseline_pref_stereo": prefs_base, "finetuned_pref_stereo": prefs_tuned}).to_csv(
        os.path.join(RESULTS_DIR, "crows_pairwise_prefs.csv"), index=False
    )

    # Detoxify evaluation
    prompts = load_prompts_jsonl(PILOT_PROMPTS)
    random.shuffle(prompts)
    prompts_sample = [p["prompt"] for p in prompts[:N_PROMPTS_DETOX]]

    print("\nGenerating responses for baseline model...")
    base_responses = batch_generate_responses(BASELINE_MODEL, prompts_sample, device=device, batch_size=8)
    base_texts = [r["response"] for r in base_responses]

    print("Generating responses for fine-tuned model...")
    tuned_responses = batch_generate_responses(FINETUNED_MODEL, prompts_sample, device=device, batch_size=8)
    tuned_texts = [r["response"] for r in tuned_responses]

    base_tox = detoxify_scores(base_texts)
    tuned_tox = detoxify_scores(tuned_texts)

    # Save responses
    pd.DataFrame(base_responses).to_json(os.path.join(RESULTS_DIR, "detox_baseline.jsonl"), orient="records", lines=True)
    pd.DataFrame(tuned_responses).to_json(os.path.join(RESULTS_DIR, "detox_finetuned.jsonl"), orient="records", lines=True)

    # Compare toxicity distributions
    mean_base = base_tox.mean()
    mean_tuned = tuned_tox.mean()
    print(f"\nDetoxify mean toxicity: baseline={mean_base:.4f}, fine-tuned={mean_tuned:.4f}")

    mean_diff, ci, p_val = bootstrap_mean_diff(base_tox, tuned_tox, n_resamples=2000)
    print(f"Mean diff (fine - base) = {mean_diff:.4f}, 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}], p_like_prob(fine<=base) = {p_val:.4f}")

    # Summary
    summary = {
        "crows_baseline_pref_stereo": float(frac_base),
        "crows_finetuned_pref_stereo": float(frac_tuned),
        "detox_mean_baseline": float(mean_base),
        "detox_mean_finetuned": float(mean_tuned),
        "detox_mean_diff": float(mean_diff),
        "detox_ci_low": float(ci[0]),
        "detox_ci_high": float(ci[1]),
        "detox_p_like_fine_le_base": float(p_val)
    }
    with open(os.path.join(RESULTS_DIR, "strong_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved results to", RESULTS_DIR)
    print("Done.")

if __name__ == "__main__":
    main()