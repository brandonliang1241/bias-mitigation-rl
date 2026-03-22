import pandas as pd
from pathlib import Path
import hashlib
import json

# --------------------
# Config
# --------------------
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/prompts")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROWS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/"
    "crows-pairs/master/data/crows_pairs_anonymized.csv"
)
CROWS_CSV = RAW_DIR / "crows_pairs.csv"
OUT_JSONL = OUT_DIR / "crows_pairwise.jsonl"

RANDOM_SEED = 42
MAX_SAMPLES = None  # set None for full dataset

# --------------------
# Load CrowS-Pairs
# --------------------
if not CROWS_CSV.exists():
    df = pd.read_csv(CROWS_URL)
    df.to_csv(CROWS_CSV, index=False)
else:
    df = pd.read_csv(CROWS_CSV)

if MAX_SAMPLES is not None:
    df = df.sample(MAX_SAMPLES, random_state=RANDOM_SEED)

# --------------------
# Helper: stable hash ID
# --------------------
def example_id(a: str, b: str) -> str:
    h = hashlib.sha256((a + b).encode("utf-8")).hexdigest()
    return f"crows_{h[:12]}"

# --------------------
# Build dataset
# --------------------
rows = []
seen = set()

for _, ex in df.iterrows():
    sent_more = ex["sent_more"].strip()
    sent_less = ex["sent_less"].strip()

    # Dedup safeguard
    key = (sent_more, sent_less)
    if key in seen:
        continue
    seen.add(key)

    rows.append({
        "id": example_id(sent_more, sent_less),
        "source": "crows_pairs",
        "task": "bias_pairwise",
        "sentence_more_biased": sent_more,
        "sentence_less_biased": sent_less,
        "label": 1,  # 1 = second sentence is less biased
        "bias_type": ex.get("bias_type", "unknown"),
        "meta": {
            "dataset": "CrowS-Pairs",
            "annotated": True
        }
    })

# --------------------
# Save
# --------------------
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved {len(rows)} examples to {OUT_JSONL}")
