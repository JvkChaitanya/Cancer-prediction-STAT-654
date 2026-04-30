"""Exploratory data-analysis figures and summary statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_cleaning import CleaningStats

sns.set_theme(style="whitegrid", context="talk")


def plot_class_distribution(y: pd.Series, output_path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    order = y.value_counts().sort_index().index
    counts = y.value_counts().sort_index().values
    bars = ax.bar(order, counts, color=sns.color_palette("viridis", len(order)),
                  edgecolor="black", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{int(h)}\n({h / len(y) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Tumor Classes", fontweight="bold")
    ax.set_xlabel("Tumor Class"); ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_sparsity(X: pd.DataFrame, output_path: Optional[Path] = None) -> plt.Figure:
    zg = (X == 0).mean()
    zs = (X == 0).mean(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax_, data, title, color in [
        (axes[0], zg, "Sparsity per Gene", "steelblue"),
        (axes[1], zs, "Sparsity per Sample", "teal"),
    ]:
        ax_.hist(data, bins=40, color=color, edgecolor="black", alpha=0.85)
        ax_.axvline(data.mean(), color="red", ls="--", lw=2,
                    label=f"mean = {data.mean():.2%}")
        ax_.set_title(title, fontweight="bold")
        ax_.set_xlabel("Fraction of Zero Values")
        ax_.legend(); ax_.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_top_variance_genes(X: pd.DataFrame, y: pd.Series, n: int = 6,
                            output_path: Optional[Path] = None) -> plt.Figure:
    top = X.var().sort_values(ascending=False).head(n).index.tolist()
    rows = [{"Gene": g, "Tumor": cls, "Expression": v}
            for g in top for cls in y.unique() for v in X.loc[y == cls, g]]
    g_ = sns.catplot(data=pd.DataFrame(rows), x="Tumor", y="Expression", col="Gene",
                     kind="box", col_wrap=3, height=3.5, aspect=1.2, palette="Set2")
    g_.set_titles("{col_name}", fontweight="bold")
    g_.fig.suptitle(f"Top {n} High-Variance Genes by Tumor Class",
                    fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        g_.fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return g_.fig


def plot_correlation_heatmap(X: pd.DataFrame, top_n: int = 20,
                             output_path: Optional[Path] = None) -> plt.Figure:
    top = X.var().sort_values(ascending=False).head(top_n).index.tolist()
    corr = X[top].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Gene Correlation Heatmap (Top {top_n} Variance Genes)",
                 fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def build_summary(df_shape, X_shape, gene_count_before: int,
                  qc: dict, stats: CleaningStats, y: pd.Series) -> dict:
    """Assemble the dict that is persisted as eda_summary.json."""
    return {
        "original_shape":              f"{df_shape[0]} samples x {gene_count_before} genes",
        "cleaned_shape":               f"{X_shape[0]} samples x {X_shape[1]} genes",
        "missing_values":              qc["missing_values"],
        "duplicate_rows":              qc["duplicate_rows"],
        "constant_genes_dropped":      len(stats.constant_genes),
        "low_variance_genes_dropped":  len(stats.low_variance_genes),
        "total_genes_removed":         len(stats.constant_genes) + len(stats.low_variance_genes),
        "matrix_sparsity":             f"{stats.sparsity:.2%}",
        "classes":                     sorted(y.unique().tolist()),
        "class_counts":                y.value_counts().to_dict(),
        "imbalance_ratio":             round(qc["imbalance_ratio"], 2),
    }
