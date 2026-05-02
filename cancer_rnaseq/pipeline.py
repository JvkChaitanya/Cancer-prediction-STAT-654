"""High-level orchestrators that compose every other module.

These four functions are the single API the Streamlit app calls. Each one runs
one stage of the project end-to-end and writes its on-disk artefacts.
"""
from __future__ import annotations
import json
from typing import Callable, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from . import (data_cleaning, data_loading, eda, feature_selection,
               modeling, pca_pipeline, tuning, visualization)
from .config import (ARTEFACTS, EDA_DIR, FEATURE_DIR, LOW_VARIANCE_THRESHOLD,
                     MODEL_DIR, PCA_VARIANCE, ROOT, TOP_K_SUBSETS,
                     feature_set_path)

# Type alias for progress callback function
ProgressFn = Optional[Callable[[float, str], None]]

# No-op progress function (does nothing)
def _noop(_: float, __: str) -> None:
    pass

# ---------------------------------------------------------------------------
# Section 1 — data cleaning + EDA
# ---------------------------------------------------------------------------
def run_data_cleaning_eda(local_dir: Optional[str] = None,
                          progress: ProgressFn = None) -> dict:
    """Load raw data, clean, generate EDA plots, save artefacts."""
    p = progress or _noop
    EDA_DIR.mkdir(exist_ok=True)
    
    # Load raw RNA-seq data
    p(0.05, "Loading raw data ...")
    features, labels = data_loading.load_raw(local_dir)
    df = data_cleaning.merge_features_labels(features, labels)
    X_all, y = data_cleaning.split_target(df)
    
    # Run quality checks
    p(0.30, "Quality checks ...")
    qc = data_cleaning.quality_report(X_all, y, df)
    
    # Remove low-variance genes
    p(0.45, "Removing constant / near-constant genes ...")
    X, stats = data_cleaning.remove_low_variance_genes(X_all, LOW_VARIANCE_THRESHOLD)
    
    # Generate EDA visualizations
    p(0.60, "Generating EDA figures ...")
    eda.plot_class_distribution(y, EDA_DIR / "01_class_distribution.png")
    eda.plot_sparsity(X, EDA_DIR / "02_sparsity_analysis.png")
    eda.plot_top_variance_genes(X, y, output_path=EDA_DIR / "03_top_variance_genes.png")
    eda.plot_correlation_heatmap(X, output_path=EDA_DIR / "04_correlation_heatmap.png")
    plt.close("all")
    
    # Save cleaned data and summary statistics
    p(0.90, "Saving artefacts ...")
    X.to_csv(ARTEFACTS["cleaned_features"])
    y.to_csv(ARTEFACTS["cleaned_labels"])
    summary = eda.build_summary(df.shape, X.shape, X_all.shape[1], qc, stats, y)
    with open(ARTEFACTS["eda_summary"], "w") as fh:
        json.dump(summary, fh, indent=2)
    
    p(1.0, "Done.")
    return summary

# ---------------------------------------------------------------------------
# Section 2 — feature engineering
# ---------------------------------------------------------------------------
def run_feature_engineering(progress: ProgressFn = None) -> dict:
    """Run ANOVA, Kruskal-Wallis, PCA, generate plots, save feature sets."""
    p = progress or _noop
    FEATURE_DIR.mkdir(exist_ok=True)
    
    # Load cleaned data
    X = pd.read_csv(ARTEFACTS["cleaned_features"], index_col=0)
    y = pd.read_csv(ARTEFACTS["cleaned_labels"], index_col=0).squeeze()
    
    # Run ANOVA F-tests with FDR correction and effect sizes
    p(0.02, "ANOVA F-tests ...")
    anova = feature_selection.compute_anova(
        X, y, progress=lambda r, msg: p(0.02 + 0.30 * r, msg))
    anova = feature_selection.apply_fdr(anova)
    anova = feature_selection.add_eta_squared(anova, X, y)
    anova.to_csv(ARTEFACTS["anova_stats"], index=False)
    
    # Run Kruskal-Wallis tests (non-parametric validation)
    p(0.40, "Kruskal-Wallis tests ...")
    kw = feature_selection.compute_kruskal_wallis(
        X, y, progress=lambda r, msg: p(0.40 + 0.20 * r, msg))
    kw.to_csv(ARTEFACTS["kw_stats"], index=False)
    
    # Compare ANOVA and KW rankings
    rho, overlap = feature_selection.ranking_agreement(anova, kw)
    merged = feature_selection.merged_for_plot(anova, kw)
    
    # Generate feature selection plots
    p(0.65, "Generating feature-engineering figures ...")
    visualization.plot_volcano(anova, FEATURE_DIR / "01_anova_volcano.png")
    visualization.plot_anova_vs_kw(merged, overlap, rho,
                                   FEATURE_DIR / "01b_anova_vs_kw.png")
    
    # Save top-K gene subsets
    p(0.78, "Saving top-K feature subsets ...")
    selected: Dict[int, List[str]] = {}
    for k in TOP_K_SUBSETS:
        top = feature_selection.select_top_k(anova, k)
        selected[k] = top
        X[top].to_csv(ROOT / f"features_top{k}.csv")
    with open(ARTEFACTS["selected_features"], "w") as fh:
        json.dump(selected, fh, indent=2)
    
    # Fit PCA to retain 95% variance
    p(0.85, "Fitting PCA ...")
    scaler, pca, X_pca = pca_pipeline.fit_scaler_pca(X, PCA_VARIANCE)
    X_scaled = scaler.transform(X)
    
    # Generate PCA visualization plots
    visualization.plot_pca_variance(pca, FEATURE_DIR / "02_pca_variance.png")
    _, pca_2d = visualization.plot_pca_2d(X_scaled, y,
                                          FEATURE_DIR / "03_pca_2d.png")
    _, pca_3d = visualization.plot_pca_3d(X_scaled, y,
                                          FEATURE_DIR / "04_pca_3d.png")
    plt.close("all")
    
    # Save PCA-transformed features
    pca_pipeline.to_pca_dataframe(X_pca, X.index).to_csv(ARTEFACTS["features_pca95"])
    pca_pipeline.save_artefacts(scaler, pca,
                                ARTEFACTS["scaler"], ARTEFACTS["pca_model"])
    
    # Build and save summary
    summary = {
        "total_genes":               X.shape[1],
        "significant_genes_fdr_005": int((anova["fdr"] < 0.05).sum()),
        "large_effect_genes_eta_014": int((anova["eta_squared"] >= 0.14).sum()),
        "top_20_genes":              selected[20],
        "pca_components_95var":      int(pca.n_components_),
        "pca_variance_pc1":          float(pca_2d.explained_variance_ratio_[0]),
        "pca_variance_pc2":          float(pca_2d.explained_variance_ratio_[1]),
        "pca_variance_pc3":          float(pca_3d.explained_variance_ratio_[2]),
        "pca_2d_total_variance":     float(pca_2d.explained_variance_ratio_.sum()),
        "pca_3d_total_variance":     float(pca_3d.explained_variance_ratio_.sum()),
        "spearman_anova_kw":         rho,
        "topk_overlap_anova_kw":     {str(k): float(v) for k, v in overlap.items()},
        "feature_sets_prepared":     list(TOP_K_SUBSETS) + ["PCA-95%"],
        "recommendation":            "use features_pca95.csv for modeling",
    }
    with open(ARTEFACTS["feature_summary"], "w") as fh:
        json.dump(summary, fh, indent=2)
    
    p(1.0, "Done.")
    return summary

# ---------------------------------------------------------------------------
# Section 3 — model training
# ---------------------------------------------------------------------------
def run_model_training(feature_choice: str, model_names: List[str],
                       cv_folds: int, progress: ProgressFn = None
                       ) -> Tuple[pd.DataFrame, dict]:
    """Stratified split → fit scaler on train → train selected models → save."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load feature set
    X = pd.read_csv(feature_set_path(feature_choice), index_col=0)
    y = pd.read_csv(ARTEFACTS["cleaned_labels"], index_col=0).squeeze()
    
    # Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = modeling.stratified_split(X, y)
    
    # Fit scaler on training data only
    scaler, X_train_s = modeling.fit_scaler(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Select models to train
    pool = modeling.make_models()
    selected = {n: pool[n] for n in model_names if n in pool}
    
    # Train all selected models with CV
    df_res, trained = modeling.train_all_models(
        selected, X_train_s, y_train, X_test_s, y_test,
        cv_folds=cv_folds, progress=progress)
    
    # Save results and generate plots
    df_res.to_csv(ARTEFACTS["model_results"], index=False)
    visualization.plot_model_comparison(df_res, MODEL_DIR / "01_model_comparison_cv.png")
    visualization.plot_metrics_comparison(df_res, MODEL_DIR / "02_model_metrics_comparison.png")
    visualization.plot_accuracy_vs_time(df_res, MODEL_DIR / "03_accuracy_vs_time.png")
    plt.close("all")
    
    # Package state for hyperparameter tuning
    state = {
        "feature_choice": feature_choice,
        "scaler":         scaler,
        "models":         trained,
        "X_train_s":      X_train_s,
        "X_test_s":       X_test_s,
        "y_train":        y_train,
        "y_test":         y_test,
        "X_test_raw":     X_test,
        "cv_folds":       cv_folds,
        "results":        df_res,
    }
    return df_res, state

# ---------------------------------------------------------------------------
# Section 3b — hyperparameter tuning
# ---------------------------------------------------------------------------
def run_hyperparameter_tuning(state: dict, progress: ProgressFn = None) -> dict:
    """Run GridSearchCV on best model, save tuned model and confusion matrix."""
    p = progress or _noop
    
    # Get best model from previous training
    best_name = state["results"].iloc[0]["Model"]
    best_test_before = float(state["results"].iloc[0]["Test_Accuracy"])
    
    # Run grid search
    p(0.1, f"GridSearchCV on {best_name} ...")
    gs, test_acc = tuning.tune(best_name,
                               state["X_train_s"], state["y_train"],
                               state["X_test_s"], state["y_test"],
                               cv_folds=state["cv_folds"])
    
    # Save tuned model and generate confusion matrix
    p(0.85, "Saving artefacts ...")
    joblib.dump(gs.best_estimator_, ARTEFACTS["best_model"])
    y_pred = gs.predict(state["X_test_s"])
    visualization.plot_confusion_matrix(
        state["y_test"], y_pred, best_name, test_acc,
        MODEL_DIR / "04_confusion_matrix_tuned.png")
    plt.close("all")
    
    # Build summary of tuning results
    out = {
        "best_model":                  best_name,
        "best_params":                 gs.best_params_,
        "best_cv_score":               float(gs.best_score_),
        "test_accuracy_before_tuning": best_test_before,
        "test_accuracy_after_tuning":  float(test_acc),
        "improvement":                 float(test_acc - best_test_before),
    }
    with open(ARTEFACTS["tuning_results"], "w") as fh:
        json.dump(out, fh, indent=2)
    
    # Update state with tuned model
    state["tuned_model"] = gs.best_estimator_
    state["tuned_pred"]  = y_pred
    
    p(1.0, "Done.")
    return out
