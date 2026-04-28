"""Raw RNA-seq data ingestion from Kaggle or a local folder."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import KAGGLE_DATASET


def load_from_kaggle() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Auto-download the TCGA RNA-seq dataset via kagglehub."""
    import kagglehub
    ds_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csvs = {p.stem.lower(): p for p in ds_dir.rglob("*.csv")}
    return pd.read_csv(csvs["data"]), pd.read_csv(csvs["labels"])


def load_from_local(folder: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data.csv + labels.csv from a local folder."""
    folder = Path(folder)
    return pd.read_csv(folder / "data.csv"), pd.read_csv(folder / "labels.csv")


def normalize_columns(features: pd.DataFrame, labels: pd.DataFrame
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rename Kaggle's anonymous index column and the `Class` label."""
    features = features.rename(columns={"Unnamed: 0": "sample_id"})
    labels = labels.rename(columns={"Unnamed: 0": "sample_id", "Class": "tumor_class"})
    return features, labels


def load_raw(local_dir: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single entry point: returns (features, labels) with normalized columns.

    Parameters
    ----------
    local_dir : str | Path | None
        If given, load data.csv + labels.csv from this folder.
        Otherwise auto-download from Kaggle.
    """
    if local_dir:
        features, labels = load_from_local(local_dir)
    else:
        features, labels = load_from_kaggle()
    return normalize_columns(features, labels)
