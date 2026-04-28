#!/usr/bin/env bash
# Headless pipeline runner: executes the full notebook end-to-end in place.
# Usage: ./run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")"

NOTEBOOK="Cancer_RNASeq_Classification.ipynb"

if [ ! -f "$NOTEBOOK" ]; then
    echo "error: $NOTEBOOK not found in $(pwd)" >&2
    exit 1
fi

echo "=> executing $NOTEBOOK end-to-end (this can take several minutes)"
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    "$NOTEBOOK"

echo "=> done. Generated artefacts:"
ls -1 \
    cleaned_features.csv cleaned_labels.csv \
    features_pca95.csv features_top20.csv features_top50.csv features_top100.csv features_top500.csv \
    anova_feature_stats.csv kw_feature_stats.csv \
    eda_summary.json feature_engineering_summary.json \
    model_comparison_results.csv hyperparameter_tuning_results.json \
    scaler.pkl pca_model.pkl best_model_tuned.pkl 2>/dev/null || true
