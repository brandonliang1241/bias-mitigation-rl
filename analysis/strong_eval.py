import os
import sys
import json
import math
import random
import tempfile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from scipy.stats import bootstrap

# config
RESULTS_DIR = "results/strong_eval"
CROWS_CSV_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
CROWS_LOCAL = "data/raw/crows_pairs.csv"
PILOT_PROMPTS = "data/prompts/pilot_prompts.jsonl" # your prompt corpus
N_PROMPTS_DETOX = 300 # number of prompts for the detoxify generation test (adjust as needed)

# models: replace with your own paths if needed
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2-0.5B-Instruct")
FINETUNED_MODEL = os.environ.get("FINETUNED_MODEL", "results/grpo_pilot/checkpoints/step_225")

# Make folders
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CROWS_LOCAL), exist_ok=True)

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
    # expected columns: sent_more (stereotypical), sent_less (anti)
    return df

def sentence_logprob(model, tokenizer, sentence, device):
    """
    Compute average log probability (per token) of a sentence using a causal LM.
    Returns average logprob (higher = more likely).
    """
    # simple whitespace normalization
    text = sentence.strip()
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # (1, seq_len, vocab)
        # compute log probs for each token given previous tokens
        # shift logits and ids
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # gather log probs of the label tokens
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1) # (1, seq_len-1)
        sum_logprob = token_log_probs.sum().item()
        avg_logprob = sum_logprob / (shift_labels.size(1) + 1e-12)
        return avg_logprob

def evaluate_crows_for_model(model_name, n_samples=1000):
    """
    For CrowS pairs: pick n_samples random pairs, compute which sentence model prefers
    (higher avg logprob). Return fraction where model prefers stereotypical sentence.
    """
    print(f"\nEvaluating CrowS-Pairs preference for model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
    model.eval()

    df = load_crows(CROWS_LOCAL)
    rows = df.sample(min(n_samples, len(df)), random_state=42)
    prefer_stereo = []
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        sent_stereo = row["sent_more"]
        sent_anti = row["sent_less"]
        # score both
        try:
            lp_st = sentence_logprob(model, tok, sent_stereo, model.device)
            lp_an = sentence_logprob(model, tok, sent_anti, model.device)
        except Exception as e:
            # fallback: treat as equal
            lp_st, lp_an = -9999.0, -9999.0
        prefer = 1 if lp_st > lp_an else 0
        prefer_stereo.append(prefer)

    frac = float(np.mean(prefer_stereo))
    return frac, prefer_stereo

# -------------------------
# Detoxify evaluation of generated responses
# -------------------------
def load_prompts_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def generate_responses_for_model(model_name, prompts, n_samples=None, greedy=True, max_new_tokens=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
    model.eval()

    out = []
    count = 0
    for ex in prompts:
        prompt_text = ex["prompt"]
        # use chat template if available
        try:
            chat = tok.apply_chat_template([{"role":"user","content":prompt_text}], tokenize=False, add_generation_prompt=True)
        except Exception:
            chat = prompt_text
        enc = tok(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            if greedy:
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
        full = tok.decode(gen[0], skip_special_tokens=True)
        response = full.split(prompt_text)[-1].strip()
        out.append({"id": ex.get("id"), "prompt": prompt_text, "response": response})
        count += 1
        if n_samples and count >= n_samples:
            break
    return out
import torch




def detoxify_scores(texts):
    """
    Computes toxicity scores for a list of texts using the detoxify library.
    Returns a list of floats.
    """
    model = Detoxify("unbiased", device="cuda" if torch.cuda.is_available() else "cpu")
    # Detoxify returns a dict of lists, e.g., {'toxicity': [...], 'severe_toxicity': [...]}
    scores = model.predict(texts)
    return scores['toxicity']


# -------------------------
# Statistical test helper
# -------------------------
def bootstrap_mean_diff(a, b, n_resamples=5000, seed=42):
    # Return bootstrap CI and p-value-ish (two-sided)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        ia = rng.choice(a, size=len(a), replace=True)
        ib = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(ib) - np.mean(ia))
    diffs = np.array(diffs)
    low, high = np.percentile(diffs, [2.5, 97.5])
    mean_diff = np.mean(diffs)
    p_val = (np.sum(diffs <= 0) + 1) / (n_resamples + 1) # probability fine_tuned <= baseline
    return mean_diff, (low, high), p_val

# -------------------------
# Main entry
# -------------------------
def main():
    print("Strong evaluation: CrowS-Pairs + Detoxify pipeline")
    download_crows(CROWS_LOCAL)
    # ---------- CrowS evaluation ----------
    frac_base, prefs_base = evaluate_crows_for_model(BASELINE_MODEL, n_samples=1000)
    frac_tuned, prefs_tuned = evaluate_crows_for_model(FINETUNED_MODEL, n_samples=1000)
    print(f"\nCrowS: baseline prefers stereotype fraction = {frac_base:.3f}")
    print(f"CrowS:  fine-tuned prefers stereotype fraction = {frac_tuned:.3f}")

    # Save CrowS results
    df_crows = pd.DataFrame({
        "baseline_pref_stereo": prefs_base,
        "finetuned_pref_stereo": prefs_tuned
    })
    df_crows.to_csv(os.path.join(RESULTS_DIR, "crows_pairwise_prefs.csv"), index=False)

    # ---------- Detoxify evaluation ----------
    prompts = load_prompts_jsonl(PILOT_PROMPTS)
    random.shuffle(prompts)
    prompts_sample = prompts[:N_PROMPTS_DETOX]

    print("\nGenerating responses for baseline model (Detoxify test)...")
    base_responses = generate_responses_for_model(BASELINE_MODEL, prompts_sample, n_samples=N_PROMPTS_DETOX, greedy=True)
    base_responses_text = [r['response'] for r in base_responses]
    base_t = np.array(detoxify_scores(base_responses_text))

    print("Generating responses for fine-tuned model...")
    tuned_responses = generate_responses_for_model(FINETUNED_MODEL, prompts_sample, n_samples=N_PROMPTS_DETOX, greedy=True)
    tuned_responses_text = [r['response'] for r in tuned_responses]
    tuned_t = np.array(detoxify_scores(tuned_responses_text))

    # Save responses
    pd.DataFrame(base_responses).to_json(os.path.join(RESULTS_DIR, "detox_baseline.jsonl"), orient="records", lines=True)
    pd.DataFrame(tuned_responses).to_json(os.path.join(RESULTS_DIR, "detox_finetuned.jsonl"), orient="records", lines=True)


    # Compare toxicity distributions
    mean_base = base_t.mean()
    mean_tuned = tuned_t.mean()
    print(f"\nDetoxify mean toxicity: baseline={mean_base:.4f}, fine-tuned={mean_tuned:.4f}")

    mean_diff, ci, p_val = bootstrap_mean_diff(base_t, tuned_t, n_resamples=2000)
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
