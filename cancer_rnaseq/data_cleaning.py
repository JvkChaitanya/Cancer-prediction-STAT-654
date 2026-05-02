"""Quality checks and constant / near-constant gene removal."""
from __future__ import annotations
from typing import List, NamedTuple, Tuple
import numpy as np
import pandas as pd
from .config import LOW_VARIANCE_THRESHOLD

class CleaningStats(NamedTuple):
    """Statistics from variance filtering: constant genes, low-variance genes, and sparsity."""
    constant_genes: List[str]
    low_variance_genes: List[str]
    sparsity: float

def merge_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Verify alignment and merge features + labels on sample_id."""
    # Critical check: Prevent silent data corruption from misaligned sample IDs
    if not (features["sample_id"].values == labels["sample_id"].values).all():
        raise ValueError("sample IDs are misaligned between features and labels")
    # Merge and set sample_id as index for efficient row access
    return features.merge(labels, on="sample_id").set_index("sample_id")

def split_target(df: pd.DataFrame, target_col: str = "tumor_class"
                 ) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from target labels (y) for sklearn compatibility."""
    # Extract all columns except the target as gene features
    gene_cols = [c for c in df.columns if c != target_col]
    return df[gene_cols], df[target_col]

def quality_report(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> dict:
    """Return missing-value, duplicate, dtype, range, and class-balance stats."""
    return {
        # Count total NaN values across all genes and samples
        "missing_values":  int(X.isna().sum().sum()),
        
        # Detect duplicate sample rows (should be 0 for clean TCGA data)
        "duplicate_rows":  int(df.duplicated().sum()),
        
        # Verify all gene columns are numeric (required for ML models)
        "all_numeric":     bool(X.dtypes.apply(lambda t: np.issubdtype(t, np.number)).all()),
        
        # Min/max expression values (sanity check for log2(RSEM+1) scale)
        "value_min":       float(X.values.min()),
        "value_max":       float(X.values.max()),
        
        # Class imbalance ratio: max_class_size / min_class_size (e.g., BRCA/COAD = 3.85)
        "imbalance_ratio": float(y.value_counts().max() / y.value_counts().min()),
    }

def remove_low_variance_genes(X: pd.DataFrame,
                              threshold: float = LOW_VARIANCE_THRESHOLD,
                              ) -> Tuple[pd.DataFrame, CleaningStats]:
    """Drop columns whose variance is exactly zero or below `threshold`.
    
    Returns the cleaned matrix and a `CleaningStats` record describing what was
    removed (so the caller can report it).
    """
    # Compute variance for each gene (column-wise)
    var = X.var()
    
    # Stage 1: Identify genes with exactly zero variance (constant across all 801 samples)
    constant = var[var == 0].index.tolist()
    
    # Stage 2: Identify genes with variance > 0 but below threshold (near-constant)
    low_var = var[(var > 0) & (var < threshold)].index.tolist()
    
    # Combine both lists for removal (set union avoids duplicates)
    drop = set(constant) | set(low_var)
    
    # Drop uninformative genes from the feature matrix
    X_clean = X.drop(columns=list(drop))
    
    # Calculate sparsity: fraction of zero values in cleaned matrix
    # Typical for RNA-Seq data (12-15% due to tissue-specific gene expression)
    sparsity = float((X_clean == 0).values.mean())
    
    # Return cleaned data + statistics for reporting in paper
    return X_clean, CleaningStats(constant, low_var, sparsity)
