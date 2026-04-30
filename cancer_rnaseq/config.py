"""Project-wide constants and artefact paths."""
from __future__ import annotations

from pathlib import Path

RNG = 42

KAGGLE_DATASET = "waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016"

LOW_VARIANCE_THRESHOLD = 0.001
PCA_VARIANCE = 0.95
TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5
TOP_K_SUBSETS = (20, 50, 100, 500)

ROOT = Path(__file__).resolve().parent.parent

EDA_DIR = ROOT / "eda_outputs"
FEATURE_DIR = ROOT / "feature_outputs"
MODEL_DIR = ROOT / "model_outputs"

ARTEFACTS = {
    "cleaned_features":  ROOT / "cleaned_features.csv",
    "cleaned_labels":    ROOT / "cleaned_labels.csv",
    "eda_summary":       ROOT / "eda_summary.json",
    "anova_stats":       ROOT / "anova_feature_stats.csv",
    "kw_stats":          ROOT / "kw_feature_stats.csv",
    "selected_features": ROOT / "selected_feature_sets.json",
    "feature_summary":   ROOT / "feature_engineering_summary.json",
    "features_pca95":    ROOT / "features_pca95.csv",
    "scaler":            ROOT / "scaler.pkl",
    "pca_model":         ROOT / "pca_model.pkl",
    "model_results":     ROOT / "model_comparison_results.csv",
    "tuning_results":    ROOT / "hyperparameter_tuning_results.json",
    "best_model":        ROOT / "best_model_tuned.pkl",
}

CLASS_PALETTE = {
    "BRCA": "#e74c3c",
    "KIRC": "#3498db",
    "LUAD": "#2ecc71",
    "PRAD": "#f39c12",
    "COAD": "#9b59b6",
}

CLASS_DESCRIPTIONS = {
    "BRCA": "Breast Invasive Carcinoma — hormone-receptor-driven epithelial malignancy.",
    "KIRC": "Kidney Renal Clear Cell Carcinoma — VHL-driven, immune-pathway active.",
    "LUAD": "Lung Adenocarcinoma — glandular lung cancer, frequently EGFR/KRAS-driven.",
    "COAD": "Colon Adenocarcinoma — WNT-pathway dominated, MSI / CIN subtypes.",
    "PRAD": "Prostate Adenocarcinoma — androgen-receptor-driven glandular cancer.",
}


def feature_set_path(name: str) -> Path:
    """Resolve a UI-visible feature-set name to its CSV path."""
    return {
        "PCA-95%": ARTEFACTS["features_pca95"],
        "Top-20":  ROOT / "features_top20.csv",
        "Top-50":  ROOT / "features_top50.csv",
        "Top-100": ROOT / "features_top100.csv",
        "Top-500": ROOT / "features_top500.csv",
    }[name]
