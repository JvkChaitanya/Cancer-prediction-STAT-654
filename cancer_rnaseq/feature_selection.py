"""Statistical feature selection: ANOVA F, BH-FDR, eta-squared, Kruskal-Wallis."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import f_oneway, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

ProgressFn = Optional[Callable[[float, str], None]]


def compute_anova(X: pd.DataFrame, y: pd.Series,
                  progress: ProgressFn = None) -> pd.DataFrame:
    """One-way ANOVA F-test per gene. Returns DataFrame with f_statistic + p_value."""
    classes = sorted(y.unique())
    n = X.shape[1]
    f_stats, p_vals = [], []
    for i, gene in enumerate(X.columns):
        groups = [X.loc[y == c, gene].values for c in classes]
        f, p = f_oneway(*groups)
        f_stats.append(f); p_vals.append(p)
        if progress and i % 2000 == 0 and i > 0:
            progress(i / n, f"ANOVA gene {i:,}/{n:,}")
    return pd.DataFrame({"gene": X.columns, "f_statistic": f_stats, "p_value": p_vals})


def apply_fdr(df: pd.DataFrame, p_col: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction; adds an `fdr` column."""
    _, fdr, _, _ = multipletests(df[p_col], alpha=alpha, method="fdr_bh")
    out = df.copy()
    out["fdr"] = fdr
    return out


def eta_squared(values: pd.Series, labels: pd.Series) -> float:
    """SS_between / SS_total — proportion of variance explained by group membership."""
    overall = values.mean()
    ss_total = ((values - overall) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(values[labels == c]) * (values[labels == c].mean() - overall) ** 2
        for c in labels.unique()
    )
    return float(ss_between / ss_total)


def add_eta_squared(anova_df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Append per-gene eta-squared and re-sort by FDR ascending."""
    out = anova_df.copy()
    out["eta_squared"] = [eta_squared(X[g], y) for g in X.columns]
    return out.sort_values("fdr").reset_index(drop=True)


def compute_kruskal_wallis(X: pd.DataFrame, y: pd.Series,
                           progress: ProgressFn = None) -> pd.DataFrame:
    """Per-gene KW H-test (non-parametric ANOVA), already FDR-corrected and sorted."""
    classes = sorted(y.unique())
    n = X.shape[1]
    h_stats, p_vals = [], []
    for i, gene in enumerate(X.columns):
        groups = [X.loc[y == c, gene].values for c in classes]
        h, p = kruskal(*groups)
        h_stats.append(h); p_vals.append(p)
        if progress and i % 2000 == 0 and i > 0:
            progress(i / n, f"KW gene {i:,}/{n:,}")
    df = pd.DataFrame({"gene": X.columns, "h_statistic": h_stats, "p_value": p_vals})
    return apply_fdr(df).sort_values("fdr").reset_index(drop=True)


def ranking_agreement(anova_df: pd.DataFrame, kw_df: pd.DataFrame,
                      ks: Tuple[int, ...] = (20, 50, 100, 500, 1000)
                      ) -> Tuple[float, Dict[int, float]]:
    """Spearman rho across all genes + top-K overlap fractions between ANOVA and KW."""
    merged = anova_df[["gene", "f_statistic"]].merge(
        kw_df[["gene", "h_statistic"]].rename(columns={"h_statistic": "h"}),
        on="gene")
    rho, _ = spearmanr(merged["f_statistic"], merged["h"])
    overlap = {k: len(set(anova_df.head(k)["gene"]) & set(kw_df.head(k)["gene"])) / k
               for k in ks}
    return float(rho), overlap


def merged_for_plot(anova_df: pd.DataFrame, kw_df: pd.DataFrame) -> pd.DataFrame:
    """Aligned table of (F, H) statistics — used by the ANOVA-vs-KW scatter plot."""
    return anova_df[["gene", "f_statistic"]].merge(
        kw_df[["gene", "h_statistic"]].rename(columns={"h_statistic": "h"}),
        on="gene")


def select_top_k(anova_df: pd.DataFrame, k: int) -> List[str]:
    """Return the top-k gene names by lowest FDR."""
    return anova_df.head(k)["gene"].tolist()
