"""StandardScaler + PCA fit / save / load utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PCA_VARIANCE, RNG


def fit_scaler_pca(X: pd.DataFrame, variance: float = PCA_VARIANCE
                   ) -> Tuple[StandardScaler, PCA, np.ndarray]:
    """Fit StandardScaler then PCA(variance) on the standardized matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=variance, random_state=RNG)
    X_pca = pca.fit_transform(X_scaled)
    return scaler, pca, X_pca


def to_pca_dataframe(X_pca: np.ndarray, index) -> pd.DataFrame:
    """Wrap a PCA matrix as a DataFrame with PC1, PC2, ... columns."""
    return pd.DataFrame(
        X_pca,
        index=index,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
    )


def save_artefacts(scaler: StandardScaler, pca: PCA,
                   scaler_path: Path, pca_path: Path) -> None:
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
