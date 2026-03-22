import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LOG_FILE = Path("results/grpo_pilot/training_log_bias.jsonl")
LOG_FILE = Path("results/grpo_pilot/training_log_balanced.jsonl")
LOG_FILE = Path("results/grpo_pilot/training_log_skew.jsonl")

# Load JSONL
df = pd.read_json(LOG_FILE, lines=True)

# Keep only needed columns
df = df[["step", "combined_reward", "kl", "mean_deberta_score"]]

# Drop rows with missing values
df = df.dropna()
df = df.sample(min(len(df), 5000))

print("Loaded rows:", len(df))
print(df.head())

# -------------------------
# Pairwise scatter plots
# -------------------------

plt.figure(figsize=(15,4))

# Reward vs KL
plt.subplot(1,3,1)
sns.scatterplot(data=df, x="kl", y="combined_reward", alpha=0.4)
plt.title("Reward vs KL")

# Reward vs DeBERTa
plt.subplot(1,3,2)
sns.scatterplot(data=df, x="mean_deberta_score", y="combined_reward", alpha=0.4)
plt.title("Reward vs DeBERTa Score")

# KL vs DeBERTa
plt.subplot(1,3,3)
sns.scatterplot(data=df, x="kl", y="mean_deberta_score", alpha=0.4)
plt.title("KL vs DeBERTa Score")

plt.tight_layout()
plt.show()

# -------------------------
# Correlation matrix
# -------------------------

plt.figure(figsize=(6,5))
sns.heatmap(
    df[["combined_reward","kl","mean_deberta_score"]].corr(),
    annot=True,
    cmap="coolwarm",
    center=0
)
plt.title("Metric Correlation")
plt.show()

df = df.groupby("step").mean().reset_index()
# fig, axes = plt.subplots(3, 1, figsize=(8,10))

# axes[0].plot(df["step"], df["combined_reward"])
# axes[0].set_title("Combined Reward vs Step")

# axes[1].plot(df["step"], df["kl"])
# axes[1].set_title("KL vs Step")

# axes[2].plot(df["step"], df["mean_deberta_score"])
# axes[2].set_title("DeBERTa Score vs Step")

# plt.tight_layout()
# plt.show()

# Rolling window size (adjust if needed)
window = 20

# Compute trend lines
df["reward_trend"] = df["combined_reward"].rolling(window).mean()
df["kl_trend"] = df["kl"].rolling(window).mean()
df["deberta_trend"] = df["mean_deberta_score"].rolling(window).mean()

fig, axes = plt.subplots(3, 1, figsize=(8,10))

# Reward plot
axes[0].plot(df["step"], df["combined_reward"], alpha=0.3, label="Raw")
axes[0].plot(df["step"], df["reward_trend"], linewidth=3, label="Trend")
axes[0].set_title("Combined Reward vs Step")
axes[0].legend()

# KL plot
axes[1].plot(df["step"], df["kl"], alpha=0.3, label="Raw")
axes[1].plot(df["step"], df["kl_trend"], linewidth=3, label="Trend")
axes[1].set_title("KL vs Step")
axes[1].legend()

# DeBERTa plot
axes[2].plot(df["step"], df["mean_deberta_score"], alpha=0.3, label="Raw")
axes[2].plot(df["step"], df["deberta_trend"], linewidth=3, label="Trend")
axes[2].set_title("DeBERTa Score vs Step")
axes[2].legend()

plt.tight_layout()
plt.show()