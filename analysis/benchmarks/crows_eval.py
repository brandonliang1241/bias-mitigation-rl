import os
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from ..utils.logprob import batch_sentence_logprob

CROWS_CSV_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
CROWS_LOCAL = "data/raw/crows_pairs.csv"

os.makedirs(os.path.dirname(CROWS_LOCAL), exist_ok=True)
def download_crows(csv_path: str, url: str):
    if Path(csv_path).exists():
        print(f"Found existing CrowS CSV at {csv_path}")
        return
    import requests
    print("Downloading CrowS-Pairs CSV...")
    r = requests.get(url)
    r.raise_for_status()
    Path(csv_path).write_bytes(r.content)
    print("Saved CrowS CSV.")

def evaluate_crows_for_model(model_name, n_samples=1000, csv_path="data/raw/crows_pairs.csv"):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    print("MODEL ID:", model.config._name_or_path) # debug
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load CrowS CSV
    df = pd.read_csv(csv_path)
    rows = df.sample(min(n_samples, len(df)), random_state=42)

    sent_stereo_list = rows["sent_more"].tolist()
    sent_anti_list = rows["sent_less"].tolist()

    # Compute log probabilities for each
    lp_stereo = batch_sentence_logprob(model, tokenizer, sent_stereo_list, device)
    lp_anti = batch_sentence_logprob(model, tokenizer, sent_anti_list, device)

    # Preference calculation
    prefer_stereo = [1 if st > an else 0 for st, an in zip(lp_stereo, lp_anti)]
    frac = float(np.mean(prefer_stereo))
    return frac, prefer_stereo

