"""Matplotlib helpers — volcano, PCA, model-comparison, confusion matrix."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import CLASS_PALETTE, RNG


# ---------------------------------------------------------------------------
# Feature-engineering plots
# ---------------------------------------------------------------------------
def plot_volcano(anova_df: pd.DataFrame, output_path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    neglog = -np.log10(anova_df["fdr"] + 1e-300)
    ax.scatter(anova_df["eta_squared"], neglog, s=8, alpha=0.3,
               color="gray", label="all genes")
    mask = (anova_df["fdr"] < 0.05) & (anova_df["eta_squared"] >= 0.14)
    ax.scatter(anova_df.loc[mask, "eta_squared"], neglog[mask], s=18, alpha=0.8,
               color="crimson", label=f"sig + large effect ({int(mask.sum()):,})")
    ax.axhline(-np.log10(0.05), ls="--", color="black", lw=1.4, alpha=0.6)
    ax.axvline(0.14, ls="--", color="black", lw=1.4, alpha=0.6)
    ax.set_xlabel("Effect Size (eta^2)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Volcano: Gene Discriminative Power", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_anova_vs_kw(merged: pd.DataFrame, overlap: dict, rho: float,
                     output_path: Optional[Path] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].scatter(np.log10(merged["f_statistic"] + 1),
                    np.log10(merged["h"] + 1),
                    s=6, alpha=0.3, color="steelblue")
    axes[0].set_xlabel("log10(F + 1)"); axes[0].set_ylabel("log10(H + 1)")
    axes[0].set_title(f"ANOVA F vs Kruskal-Wallis H  (rho={rho:.3f})", fontweight="bold")
    axes[0].grid(alpha=0.3)

    ks = list(overlap.keys())
    axes[1].plot(ks, [overlap[k] * 100 for k in ks], "o-",
                 color="darkorange", lw=2.5, markersize=10)
    axes[1].set_xlabel("Top-K"); axes[1].set_ylabel("Overlap (%)")
    axes[1].set_title("Top-K overlap: ANOVA vs KW", fontweight="bold")
    axes[1].set_ylim([0, 100]); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_pca_variance(pca: PCA, output_path: Optional[Path] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[0].plot(range(1, len(cumsum) + 1), cumsum, lw=2.5)
    axes[0].axhline(0.95, color="red", ls="--", lw=2, label="95%")
    axes[0].set_xlabel("Components"); axes[0].set_ylabel("Cumulative variance")
    axes[0].set_title("Cumulative Explained Variance", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(range(1, 21), pca.explained_variance_ratio_[:20],
                color="teal", edgecolor="black")
    axes[1].set_xlabel("Component"); axes[1].set_ylabel("Variance ratio")
    axes[1].set_title("Top 20 Components", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_pca_2d(X_scaled: np.ndarray, y: pd.Series,
                output_path: Optional[Path] = None) -> Tuple[plt.Figure, PCA]:
    """Fit a 2-component PCA on `X_scaled` and scatter, coloured by class."""
    pca_2d = PCA(n_components=2, random_state=RNG)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cls in sorted(y.unique()):
        m = (y == cls).values
        ax.scatter(X_2d[m, 0], X_2d[m, 1], label=cls, alpha=0.7, s=55,
                   color=CLASS_PALETTE.get(cls, "gray"),
                   edgecolors="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA 2D Projection", fontweight="bold")
    ax.legend(title="Tumor"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig, pca_2d


def plot_pca_3d(X_scaled: np.ndarray, y: pd.Series,
                output_path: Optional[Path] = None) -> Tuple[plt.Figure, PCA]:
    """Fit a 3-component PCA on `X_scaled` and scatter in 3D, coloured by class."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    pca_3d = PCA(n_components=3, random_state=RNG)
    X_3d = pca_3d.fit_transform(X_scaled)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for cls in sorted(y.unique()):
        m = (y == cls).values
        ax.scatter(X_3d[m, 0], X_3d[m, 1], X_3d[m, 2], label=cls, alpha=0.7, s=45,
                   color=CLASS_PALETTE.get(cls, "gray"),
                   edgecolors="black", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]*100:.2f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]*100:.2f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]*100:.2f}%)", labelpad=10)
    total_var = pca_3d.explained_variance_ratio_.sum() * 100
    ax.set_title(f"PCA 3D Projection  (total: {total_var:.2f}%)",
                 fontweight="bold", pad=18)
    ax.legend(title="Tumor", loc="upper left")
    ax.view_init(elev=20, azim=45)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig, pca_3d


# ---------------------------------------------------------------------------
# Model-comparison plots
# ---------------------------------------------------------------------------
def plot_model_comparison(results: pd.DataFrame,
                          output_path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6))
    sorted_r = results.sort_values("CV_Accuracy_Mean")
    ax.barh(sorted_r["Model"], sorted_r["CV_Accuracy_Mean"],
            xerr=sorted_r["CV_Accuracy_Std"], color="steelblue",
            edgecolor="black", alpha=0.85, capsize=5)
    for i, (_, row) in enumerate(sorted_r.iterrows()):
        ax.text(row["CV_Accuracy_Mean"] + 0.01, i,
                f"{row['CV_Accuracy_Mean']:.3f}", va="center", fontweight="bold")
    ax.set_xlabel("CV Accuracy (5-fold)"); ax.set_xlim([0, 1.05])
    ax.set_title("Model Comparison — CV Accuracy", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_metrics_comparison(results: pd.DataFrame,
                            output_path: Optional[Path] = None) -> plt.Figure:
    metrics = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1"]
    fig, ax = plt.subplots(figsize=(15, 6))
    xs = np.arange(len(results))
    w = 0.2
    colors = ["steelblue", "orange", "green", "crimson"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(xs + i * w, results[m], w, label=m.replace("Test_", ""),
               color=c, alpha=0.85, edgecolor="black")
    ax.set_xticks(xs + w * 1.5)
    ax.set_xticklabels(results["Model"], rotation=35, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim([0, 1.05])
    ax.set_title("Test-Set Metrics Across Models", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_accuracy_vs_time(results: pd.DataFrame,
                          output_path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(results["Training_Time"], results["CV_Accuracy_Mean"],
               s=240, c=range(len(results)), cmap="viridis",
               alpha=0.85, edgecolors="black", linewidth=2)
    for _, row in results.iterrows():
        ax.annotate(row["Model"], (row["Training_Time"], row["CV_Accuracy_Mean"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    ax.set_xlabel("Training Time (s)"); ax.set_ylabel("CV Accuracy")
    ax.set_title("Model Efficiency: Accuracy vs Training Time", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name: str, accuracy: float,
                          output_path: Optional[Path] = None) -> plt.Figure:
    labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name} (tuned)\nTest accuracy {accuracy:.4f}",
                 fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
    return fig
