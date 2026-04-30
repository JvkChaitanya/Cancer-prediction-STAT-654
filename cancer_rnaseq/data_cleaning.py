"""Quality checks and constant / near-constant gene removal."""
from __future__ import annotations

from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd

from .config import LOW_VARIANCE_THRESHOLD


class CleaningStats(NamedTuple):
    constant_genes: List[str]
    low_variance_genes: List[str]
    sparsity: float


def merge_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Verify alignment and merge features + labels on sample_id."""
    if not (features["sample_id"].values == labels["sample_id"].values).all():
        raise ValueError("sample IDs are misaligned between features and labels")
    return features.merge(labels, on="sample_id").set_index("sample_id")


def split_target(df: pd.DataFrame, target_col: str = "tumor_class"
                 ) -> Tuple[pd.DataFrame, pd.Series]:
    gene_cols = [c for c in df.columns if c != target_col]
    return df[gene_cols], df[target_col]


def quality_report(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> dict:
    """Return missing-value, duplicate, dtype, range, and class-balance stats."""
    return {
        "missing_values":  int(X.isna().sum().sum()),
        "duplicate_rows":  int(df.duplicated().sum()),
        "all_numeric":     bool(X.dtypes.apply(lambda t: np.issubdtype(t, np.number)).all()),
        "value_min":       float(X.values.min()),
        "value_max":       float(X.values.max()),
        "imbalance_ratio": float(y.value_counts().max() / y.value_counts().min()),
    }


def remove_low_variance_genes(X: pd.DataFrame,
                              threshold: float = LOW_VARIANCE_THRESHOLD,
                              ) -> Tuple[pd.DataFrame, CleaningStats]:
    """Drop columns whose variance is exactly zero or below `threshold`.

    Returns the cleaned matrix and a `CleaningStats` record describing what was
    removed (so the caller can report it).
    """
    var = X.var()
    constant = var[var == 0].index.tolist()
    low_var = var[(var > 0) & (var < threshold)].index.tolist()
    drop = set(constant) | set(low_var)
    X_clean = X.drop(columns=list(drop))
    sparsity = float((X_clean == 0).values.mean())
    return X_clean, CleaningStats(constant, low_var, sparsity)
