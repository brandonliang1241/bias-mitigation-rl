import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LOG_FILES = {
    "Bias": Path("results/grpo_pilot/training_log_bias.jsonl"),
    "Balanced": Path("results/grpo_pilot/training_log_balanced.jsonl"),
    "Skew": Path("results/grpo_pilot/training_log_skew.jsonl"),
    "Bias+": Path("results/grpo_pilot/training_log_bias_1800.jsonl"),
}

dfs = {}

# -------------------------
# Load data
# -------------------------

for label, file in LOG_FILES.items():

    df = pd.read_json(file, lines=True)

    df = df[["step", "combined_reward", "kl", "mean_deberta_score"]].dropna()

    # sample scatter points (keeps plots readable)
    df_sample = df.sample(min(len(df), 5000))

    dfs[label] = {
        "raw": df,
        "scatter": df_sample
    }

# -------------------------
# Scatter plots (original diagnostics)
# -------------------------

# fig, axes = plt.subplots(3, 3, figsize=(15,10))

# for col, (label, data) in enumerate(dfs.items()):

#     df = data["scatter"]

#     sns.scatterplot(
#         data=df, x="kl", y="combined_reward",
#         alpha=0.3, ax=axes[0, col]
#     )
#     axes[0, col].set_title(f"{label}: Reward vs KL")

#     sns.scatterplot(
#         data=df, x="mean_deberta_score", y="combined_reward",
#         alpha=0.3, ax=axes[1, col]
#     )
#     axes[1, col].set_title(f"{label}: Reward vs DeBERTa")

#     sns.scatterplot(
#         data=df, x="kl", y="mean_deberta_score",
#         alpha=0.3, ax=axes[2, col]
#     )
#     axes[2, col].set_title(f"{label}: KL vs DeBERTa")

# plt.tight_layout()
# plt.show()

# -------------------------
# Training curves comparison
# -------------------------

plt.figure(figsize=(10,6))

for label, data in dfs.items():

    df = data["raw"]

    df = df.groupby("step").mean(numeric_only=True).reset_index()

    df["trend"] = df["combined_reward"].rolling(20).mean()

    plt.plot(df["step"], df["trend"], label=label)

plt.title("Combined Reward vs Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.show()

# -------------------------
# KL comparison
# -------------------------

plt.figure(figsize=(10,6))

for label, data in dfs.items():

    df = data["raw"]

    df = df.groupby("step").mean(numeric_only=True).reset_index()

    df["trend"] = df["kl"].rolling(20).mean()

    plt.plot(df["step"], df["trend"], label=label)

plt.title("KL vs Step")
plt.xlabel("Step")
plt.ylabel("KL")
plt.legend()
plt.show()

# -------------------------
# DeBERTa comparison
# -------------------------

plt.figure(figsize=(10,6))

for label, data in dfs.items():

    df = data["raw"]

    df = df.groupby("step").mean(numeric_only=True).reset_index()

    df["trend"] = -df["mean_deberta_score"].rolling(20).mean()

    plt.plot(df["step"], df["trend"], label=label)

plt.title("DeBERTa Score vs Step")
plt.xlabel("Step")
plt.ylabel("DeBERTa Score")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))

# -------------------------
# Combined_reward vs KL comparison
# -------------------------
window = 20

for label, data in dfs.items():

    df = data["raw"]

    df = df.groupby("step").mean(numeric_only=True).reset_index()

    # smoothing
    df["kl_smooth"] = df["kl"].rolling(window).mean()
    df["reward_smooth"] = df["combined_reward"].rolling(window).mean()
    plt.plot(df["kl_smooth"], df["reward_smooth"], label=label)

plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("Reward vs KL (Training Trajectory)")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))

# -------------------------
# Reward vs DeBERTa comparison
# -------------------------

for label, data in dfs.items():

    df = data["raw"]

    df = df.groupby("step").mean(numeric_only=True).reset_index()

    df["deberta_smooth"] = -df["mean_deberta_score"].rolling(window).mean()
    df["reward_smooth"] = df["combined_reward"].rolling(window).mean()

    plt.plot(df["deberta_smooth"], df["reward_smooth"], label=label)

plt.xlabel("DeBERTa Score")
plt.ylabel("Reward")
plt.title("Reward vs DeBERTa Score")
plt.legend()
plt.show()