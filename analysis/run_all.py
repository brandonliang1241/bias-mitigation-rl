import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

# python -m analysis.run_all

# Import benchmarks inside the functions

# -------------------------
# Config
# -------------------------
RESULTS_DIR = "results/strong_eval"
CROWS_CSV_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
CROWS_LOCAL = "data/raw/crows_pairs.csv"
PILOT_PROMPTS = "data/prompts/pilot_prompts.jsonl"

BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2-0.5B-Instruct")
FINETUNED_MODEL = os.environ.get("FINETUNED_MODEL", "results/grpo_pilot/checkpoints/step_700")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CROWS_LOCAL), exist_ok=True)

# -------------------------
# Benchmark registry
# -------------------------
def run_crows():
    from analysis.benchmarks.crows_eval import evaluate_crows_for_model, download_crows
    download_crows(CROWS_LOCAL, CROWS_CSV_URL)
    frac_base, prefs_base = evaluate_crows_for_model(BASELINE_MODEL, n_samples=1000, csv_path=CROWS_LOCAL, is_lora = False)
    frac_tuned, prefs_tuned = evaluate_crows_for_model(FINETUNED_MODEL, n_samples=1000, csv_path=CROWS_LOCAL, is_lora = True)

    pd.DataFrame({
        "baseline_pref_stereo": prefs_base,
        "finetuned_pref_stereo": prefs_tuned
    }).to_csv(os.path.join(RESULTS_DIR, "crows_pairwise_prefs.csv"), index=False)
    
    prefs_base = np.array(prefs_base)
    prefs_tuned = np.array(prefs_tuned)

    print("Num flipped decisions:", np.sum(prefs_base != prefs_tuned))
    print("Total samples:", len(prefs_base))

    return {
        "crows_baseline_pref_stereo": float(frac_base),
        "crows_finetuned_pref_stereo": float(frac_tuned),
    }

def run_detoxify():
    from analysis.benchmarks.detox_eval import evaluate_detoxify
    mean_base, mean_tuned, mean_diff, ci, p_val = evaluate_detoxify(
        BASELINE_MODEL, FINETUNED_MODEL, PILOT_PROMPTS, device="cuda"
    )
    return {
        "detox_mean_baseline": float(mean_base),
        "detox_mean_finetuned": float(mean_tuned),
        "detox_mean_diff": float(mean_diff),
        "detox_ci_low": float(ci[0]),
        "detox_ci_high": float(ci[1]),
        "detox_p_like_fine_le_base": float(p_val),
    }

def run_truthfulqa():
    from analysis.benchmarks.truthfulqa_eval import evaluate_truthfulqa
    return evaluate_truthfulqa(BASELINE_MODEL, FINETUNED_MODEL, split="validation")

# SPECIFY BENCHMARKS HERE
BENCHMARKS = {
    "crows": run_crows,
    # "detoxify": run_detoxify,
    # "truthfulqa": run_truthfulqa,
}

# -------------------------
# Main
# -------------------------
def main():
    print("Running strong evaluation benchmarks…")

    print("Baseline model:", BASELINE_MODEL)
    print("Finetuned model:", FINETUNED_MODEL)

    results = {}
    for name, fn in BENCHMARKS.items():
        print(f"\n>>> Running {name} evaluation")
        res = fn()
        results.update(res)

    # Save summary JSON
    out_path = os.path.join(RESULTS_DIR, "strong_eval_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to", out_path)
    print("Done.")


if __name__ == "__main__":
    main()
