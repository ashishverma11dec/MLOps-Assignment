from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.data_pipeline import load_processed_dataset
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.data_pipeline import load_processed_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"


def generate_eda() -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_dataset()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, column in enumerate(["age", "chol", "thalach", "oldpeak"]):
        sns.histplot(df[column], bins=20, ax=axes[idx], kde=True)
        axes[idx].set_title(f"Distribution of {column}")

    balance_path = OUTPUT_DIR / "class_balance.png"
    plt.tight_layout()
    fig.savefig(balance_path, dpi=200)
    plt.close(fig)

    corr = df.corr(numeric_only=True)
    heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    class_counts = df["target"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    class_counts.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title("Class Balance")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No Disease", "Disease"], rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance_summary.png", dpi=200)
    plt.close(fig)

    return str(OUTPUT_DIR)


if __name__ == "__main__":
    output = generate_eda()
    print(f"EDA plots saved to {output}")
