"""Cancer RNA-Seq classification package.

A modular pipeline for classifying TCGA Pan-Cancer RNA-seq samples into
five tumour types (BRCA, KIRC, LUAD, COAD, PRAD).

Sub-modules
-----------
config             — project-wide constants and artefact paths
data_loading       — Kaggle / local raw data ingestion
data_cleaning      — quality checks and constant-gene removal
eda                — exploratory figures and summary stats
feature_selection  — ANOVA F, BH-FDR, eta-squared, Kruskal-Wallis
pca_pipeline       — StandardScaler + PCA fit / save
visualization      — matplotlib helpers (volcano, PCA, comparison, CM)
modeling           — model factory, stratified split, training loop
tuning             — hyperparameter grids and GridSearchCV runner
pipeline           — high-level orchestrators used by the notebook + app
"""

__all__ = [
    "config",
    "data_loading",
    "data_cleaning",
    "eda",
    "feature_selection",
    "pca_pipeline",
    "visualization",
    "modeling",
    "tuning",
    "pipeline",
]
