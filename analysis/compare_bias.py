import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

def load_results():
    """Load all CSVs from results/ into a dict of DataFrames."""
    dfs = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".csv"):
            path = os.path.join(RESULTS_DIR, fname)
            dfs[fname.replace(".csv", "")] = pd.read_csv(path)
    return dfs

def compare_models(dfs):
    """Compute average reward overall and per bias type."""
    summary = {}
    for name, df in dfs.items():
        overall = df["reward_bias"].mean()
        by_type = df.groupby("bias_type")["reward_bias"].mean()
        summary[name] = {"overall": overall, "by_type": by_type}
    return summary

def plot_by_type(summary):
    """Plot per-bias-type comparison as grouped bar chart."""
    # Collect all bias types across models
    all_bias_types = sorted(set().union(*[s["by_type"].index for s in summary.values()]))

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        model: [summary[model]["by_type"].get(bt, float("nan")) for bt in all_bias_types]
        for model in summary
    }, index=all_bias_types)

    # Plot
    ax = plot_data.plot(kind="bar", figsize=(10,6))
    plt.title("Per-Bias-Type Average Reward (Higher = Less Biased)")
    plt.xlabel("Bias Type")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

def main():
    dfs = load_results()
    if not dfs:
        print("No results found in results/ directory.")
        return
    
    summary = compare_models(dfs)

    # Print textual summary
    print("=== Overall Bias Reward Averages ===")
    for model, stats in summary.items():
        print(f"{model}: {stats['overall']:.3f}")

    print("\n=== Per Bias Type ===")
    for model, stats in summary.items():
        print(f"\n{model}:")
        print(stats["by_type"])

    # Plot
    plot_by_type(summary)

if __name__ == "__main__":
    main()
