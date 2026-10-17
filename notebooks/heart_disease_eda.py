from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_pipeline import load_processed_dataset

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "notebook_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_processed_dataset()
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feature in zip(axes.flat, ["age", "chol", "thalach", "oldpeak"]):
        sns.histplot(df[feature], bins=20, kde=True, ax=ax)
        ax.set_title(f"{feature} distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=200)
    plt.close(fig)

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=200)
    plt.close()

    class_counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["No disease", "Disease"], class_counts.values, color=["#5DA5DA", "#FAA43A"])
    ax.set_title("Target distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=200)
    plt.close(fig)

    print(f"Notebook EDA outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
