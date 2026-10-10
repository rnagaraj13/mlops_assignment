"""
Exploratory Data Analysis – Heart Disease UCI Dataset.
Run:  python notebooks/eda.py
Outputs plots to notebooks/plots/
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import (CATEGORICAL_COLS, FEATURE_COLS,
                                    NUMERICAL_COLS, TARGET_COL, load_data)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")

sns.set_theme(style="whitegrid", palette="Set2")


def save(fig, name):
    p = os.path.join(PLOT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def main():
    df = load_data(DATA_PATH)
    print("=== Dataset Overview ===")
    print(df.shape)
    print(df.dtypes)
    print(df.describe())

    # ── 1. Class Balance ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET_COL].value_counts()
    bars = ax.bar(
        ["No Disease (0)", "Heart Disease (1)"],
        counts.values,
        color=["#5DA5DA", "#F15854"],
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 2,
            str(v),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Class Distribution (Heart Disease)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    save(fig, "01_class_balance.png")

    # ── 2. Numerical feature histograms ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, col in enumerate(NUMERICAL_COLS):
        for label, grp in df.groupby(TARGET_COL)[col]:
            axes[i].hist(
                grp, bins=20, alpha=0.65, label="Disease" if label else "No Disease"
            )
        axes[i].set_title(col, fontsize=12)
        axes[i].legend(fontsize=9)
    axes[-1].axis("off")
    fig.suptitle(
        "Numerical Feature Distributions by Class", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "02_numerical_histograms.png")

    # ── 3. Correlation heatmap ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_correlation_heatmap.png")

    # ── 4. Categorical feature counts ────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    for i, col in enumerate(CATEGORICAL_COLS):
        cross = pd.crosstab(df[col], df[TARGET_COL])
        cross.plot(
            kind="bar",
            ax=axes[i],
            color=["#5DA5DA", "#F15854"],
            edgecolor="white",
            linewidth=0.8,
        )
        axes[i].set_title(col, fontsize=12)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=0)
        axes[i].legend(["No Disease", "Disease"], fontsize=8)
    plt.suptitle("Categorical Features vs Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "04_categorical_vs_target.png")

    # ── 5. Box plots numerical vs target ─────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for i, col in enumerate(NUMERICAL_COLS):
        df.boxplot(
            column=col,
            by=TARGET_COL,
            ax=axes[i],
            boxprops=dict(color="#5DA5DA"),
            medianprops=dict(color="#F15854", linewidth=2),
        )
        axes[i].set_title(col)
        axes[i].set_xlabel("Target (0=No, 1=Disease)")
    plt.suptitle("Numerical Features by Target Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_boxplots.png")

    print("\nEDA complete. Plots saved to", PLOT_DIR)


if __name__ == "__main__":
    main()
