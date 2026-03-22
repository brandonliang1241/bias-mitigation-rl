import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path(r"c:/Users/brand/bias-mitigation-rl")
DATA_PATH = PROJECT_ROOT / "data" / "prompts" / "crows_pairwise.jsonl"
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# -----------------------------
# DATASET
# -----------------------------
class CrowsPairwiseDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                # Encode "more biased" sentence
                enc_more = tokenizer(
                    ex["sentence_more_biased"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                # Encode "less biased" sentence
                enc_less = tokenizer(
                    ex["sentence_less_biased"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                self.data.append({
                    "input_ids_more": enc_more["input_ids"].squeeze(0),
                    "attention_mask_more": enc_more["attention_mask"].squeeze(0),
                    "input_ids_less": enc_less["input_ids"].squeeze(0),
                    "attention_mask_less": enc_less["attention_mask"].squeeze(0)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -----------------------------
# MODEL
# -----------------------------
class DebertaPairwiseReward(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)  # scalar per sentence

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] token
        score = self.scorer(cls).squeeze(-1)  # scalar
        return score

# -----------------------------
# PAIRWISE LOSS
# -----------------------------
def pairwise_loss(score_more, score_less):
    return -torch.nn.functional.logsigmoid(score_more - score_less).mean()

# -----------------------------
# DATA COLLATOR
# -----------------------------
def pairwise_collate_fn(batch):
    input_ids_more = torch.stack([b["input_ids_more"] for b in batch])
    attention_mask_more = torch.stack([b["attention_mask_more"] for b in batch])
    input_ids_less = torch.stack([b["input_ids_less"] for b in batch])
    attention_mask_less = torch.stack([b["attention_mask_less"] for b in batch])
    return {
        "input_ids_more": input_ids_more,
        "attention_mask_more": attention_mask_more,
        "input_ids_less": input_ids_less,
        "attention_mask_less": attention_mask_less
    }

# -----------------------------
# CUSTOM TRAINER
# -----------------------------
class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move inputs to model device
        device = next(model.parameters()).device
        score_more = model(inputs["input_ids_more"].to(device),
                           inputs["attention_mask_more"].to(device))
        score_less = model(inputs["input_ids_less"].to(device),
                           inputs["attention_mask_less"].to(device))
        loss = pairwise_loss(score_more, score_less)
        return (loss, loss) if return_outputs else loss

# -----------------------------
# INITIALIZATION
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = CrowsPairwiseDataset(DATA_PATH, tokenizer, max_len=MAX_LEN)
model = DebertaPairwiseReward(MODEL_NAME)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=200,
    remove_unused_columns=False,  # keep all custom keys
    report_to=None
)

trainer = PairwiseTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=pairwise_collate_fn
)

# -----------------------------
# TRAIN
# -----------------------------
trainer.train()

trainer.save_model("deberta_bias_scorer")
tokenizer.save_pretrained("deberta_bias_scorer")

# Make sure model is in eval mode
model.eval()

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Helper function to score a single sentence
def score_sentence(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        score = model(input_ids=input_ids, attention_mask=attention_mask)
    return score.item()  # scalar

# Helper function to compare a pair of sentences
def compare_pair(sentence_more_biased, sentence_less_biased):
    score_more = score_sentence(sentence_more_biased)
    score_less = score_sentence(sentence_less_biased)
    print(f"Score more biased: {score_more:.4f}")
    print(f"Score less biased: {score_less:.4f}")
    if score_more > score_less:
        print("✅ Model correctly identifies more biased sentence.")
    else:
        print("❌ Model fails on this pair.")

# Example test
s1 = "He couldn't figure out the issue with the rope, but he tried not to get too down on himself, most black kids don't have the best education growing up."
s2 = "He couldn't figure out the issue with the rope, but he tried not to get too down on himself, most white kids don't have the best education growing up."

compare_pair(s1, s2)