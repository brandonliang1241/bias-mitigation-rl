import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-small"
MODEL_DIR = "deberta_bias_scorer"
PROJECT_ROOT = Path(r"c:/Users/brand/bias-mitigation-rl")
DATA_PATH = PROJECT_ROOT / "data" / "prompts" / "crows_pairwise.jsonl"
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


class DebertaPairwiseReward(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        score = self.scorer(cls).squeeze(-1)
        return score

def evaluate_pairwise_accuracy(
    model,
    tokenizer,
    data_path,
    max_len=128,
    device=None
):
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            ex = json.loads(line)

            s_more = ex["sentence_more_biased"]
            s_less = ex["sentence_less_biased"]

            enc_more = tokenizer(
                s_more,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            enc_less = tokenizer(
                s_less,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )

            with torch.no_grad():
                score_more = model(
                    enc_more["input_ids"].to(device),
                    enc_more["attention_mask"].to(device)
                )
                score_less = model(
                    enc_less["input_ids"].to(device),
                    enc_less["attention_mask"].to(device)
                )

            if score_more.item() > score_less.item():
                correct += 1
            total += 1

    acc = correct / total
    print(f"\nPairwise accuracy: {acc:.4f} ({correct}/{total})")
    return acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DebertaPairwiseReward(MODEL_NAME)
from safetensors.torch import load_file

state_dict = load_file(f"{MODEL_DIR}/model.safetensors")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

acc = evaluate_pairwise_accuracy(
    model=model,
    tokenizer=tokenizer,
    data_path=DATA_PATH,
    max_len=MAX_LEN
)

# Test 
text = "Black people are not lazy."
enc = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
).to(device)

scores = []

model.eval()
with torch.no_grad():
    for _ in range(100):
        score = model(
            enc["input_ids"],
            enc["attention_mask"]
        ).item()
        scores.append(score)

print(f"min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")

# deltas = []

# with open(DATA_PATH, "r", encoding="utf-8") as f:
#     for line in f:
#         ex = json.loads(line)

#         enc_more = tokenizer(
#             ex["sentence_more_biased"],
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=MAX_LEN
#         ).to(device)

#         enc_less = tokenizer(
#             ex["sentence_less_biased"],
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=MAX_LEN
#         ).to(device)

#         with torch.no_grad():
#             s_more = model(enc_more["input_ids"], enc_more["attention_mask"]).item()
#             s_less = model(enc_less["input_ids"], enc_less["attention_mask"]).item()

#         deltas.append(s_more - s_less)

# import numpy as np
# print(
#     "min:", np.min(deltas),
#     "mean:", np.mean(deltas),
#     "max:", np.max(deltas),
#     "pct_positive:", np.mean(np.array(deltas) > 0)
# )