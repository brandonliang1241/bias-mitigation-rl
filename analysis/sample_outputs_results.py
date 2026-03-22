import json
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# =====================
# CONFIG
# =====================

JSON_FILE = Path("data/model_comparisons_bias_1800/side_by_side_gpt_judge.json")

LABELS = [
    "no difference",
    "baseline better",
    "finetuned better",
    "mixed"
]

# =====================
# HELPERS
# =====================

def normalize_label(label: str) -> str:
    """
    Normalize GPT outputs like "\"baseline better\"" -> "baseline better"
    """
    if label is None:
        return "unknown"

    label = label.strip().lower()
    label = label.replace('"', "").replace("\\", "")
    return label

# =====================
# LOAD DATA
# =====================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

human_labels = []
gpt_labels = []

for entry in data:
    human = normalize_label(entry.get("improvement"))
    gpt = normalize_label(entry.get("gpt_judge"))

    if human in LABELS and gpt in LABELS:
        human_labels.append(human)
        gpt_labels.append(gpt)

print(f"Loaded {len(human_labels)} comparable samples")

# =====================
# LABEL COUNTS
# =====================

print("\n📊 Manual label counts:")
for k, v in Counter(human_labels).items():
    print(f"{k}: {v}")

print("\n🤖 GPT-judge label counts:")
for k, v in Counter(gpt_labels).items():
    print(f"{k}: {v}")

# =====================
# PER-CLASS AGREEMENT TABLE
# =====================

cm = confusion_matrix(
    human_labels,
    gpt_labels,
    labels=LABELS
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Human: {l}" for l in LABELS],
    columns=[f"GPT: {l}" for l in LABELS]
)

print("\n📋 Per-class agreement table:")
print(cm_df)

# =====================
# COHEN'S KAPPA
# =====================

kappa = cohen_kappa_score(human_labels, gpt_labels, labels=LABELS)

print(f"\n🤝 Cohen’s Kappa (4-class): {kappa:.3f}")

# =====================
# Label Collapse
# =====================
def collapse_label(label):
    if label in ["no difference", "mixed"]:
        return "tie"
    return label

collapsed_human = [collapse_label(l) for l in human_labels]
collapsed_gpt = [collapse_label(l) for l in gpt_labels]

COLLAPSED_LABELS = ["baseline better", "finetuned better", "tie"]

kappa_collapsed = cohen_kappa_score(
    collapsed_human,
    collapsed_gpt,
    labels=COLLAPSED_LABELS
)

print(f"\n🤝 Cohen’s Kappa (collapsed 3-class): {kappa_collapsed:.3f}")
