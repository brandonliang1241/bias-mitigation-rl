import os
import subprocess
import pandas as pd
import re
import csv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from analysis.utils.gen import batch_generate_responses  # your helper

# -------------------------
# Config
# -------------------------
BASELINE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
FINETUNED_MODEL = "results/grpo_pilot/checkpoints/step_1000"
TRUTHFULQA_CSV = "TruthfulQA.csv"          # Original dataset
OUTPUT_CSV = "truthfulqa_input.csv"        # CSV with model answers for evaluation
SUMMARY_CSV = "summary.csv"
MAX_SAMPLES = 50
SPLIT = "validation"
DEVICE = 0  # GPU index

# -------------------------
# Helper to sanitize answers
# -------------------------
def sanitize_answer(text: str) -> str:
    if not text:
        return "N/A"
    text = re.sub(r'^\s*assistant\s*[:\n]?', '', text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    text = text.encode("ascii", errors="ignore").decode()
    return text[:2000]

# -------------------------
# Load TruthfulQA dataset
# -------------------------
df = pd.read_csv(TRUTHFULQA_CSV)
df = df.head(MAX_SAMPLES)  # optional: limit for testing
questions = df["Question"].tolist()

# -------------------------
# Generate answers from models
# -------------------------
def generate_answers(model_name, questions):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(f"cuda:{DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    answers = batch_generate_responses(model, tokenizer, questions, device=f"cuda:{DEVICE}")
    return [sanitize_answer(a) for a in answers]

print("Generating baseline model answers...")
baseline_answers = generate_answers(BASELINE_MODEL, questions)
print("Generating finetuned model answers...")
finetuned_answers = generate_answers(FINETUNED_MODEL, questions)

# -------------------------
# Add model answers to the dataframe
# -------------------------
df["baseline_model::answer"] = baseline_answers
df["finetuned_model::answer"] = finetuned_answers

# -------------------------
# Save CSV for TruthfulQA evaluation
# -------------------------
df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
print(f"CSV with model answers saved to {OUTPUT_CSV}")

# -------------------------
# Call TruthfulQA evaluate.py
# -------------------------
res = subprocess.run([
    "C:/Users/brand/bias-mitigation-rl/.venv/Scripts/python.exe",
    "evaluate.py",
    "--input_path", os.path.abspath(OUTPUT_CSV),
    "--output_path", os.path.abspath(SUMMARY_CSV),
    "--models", "baseline_model, finetuned_model",
    "--metrics", "mc bleu bleurt judge",
    "--preset", "qa",
    "--device", str(DEVICE),
], cwd="C:/Users/brand/bias-mitigation-rl/TruthfulQA/truthfulqa",
   capture_output=True, text=True
)

print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)

if res.returncode != 0:
    raise RuntimeError("evaluate.py failed, see STDERR above")

# -------------------------
# Load results CSV
# -------------------------
results_df = pd.read_csv(SUMMARY_CSV, index_col=0)
results = results_df.to_dict(orient="index")  # {Metric: {Model: Value}}
print(f"Results loaded from {SUMMARY_CSV}")

