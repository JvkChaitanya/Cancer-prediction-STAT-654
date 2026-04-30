"""Streamlit UI for the Cancer RNA-Seq classifier.

This file is intentionally kept thin: it only handles widgets, layout, and
session state. All ML / pipeline logic lives in the `cancer_rnaseq` package.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

# Allow `streamlit run app.py` to find the package whatever cwd Streamlit picked
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cancer_rnaseq import pipeline
from cancer_rnaseq.config import (ARTEFACTS, CLASS_DESCRIPTIONS, ROOT,
                                  feature_set_path)
from cancer_rnaseq.modeling import make_models
from cancer_rnaseq.tuning import PARAM_GRIDS

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cancer RNA-Seq Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _progress_callback(bar):
    return lambda p, msg: bar.progress(p, text=msg)


# ---------------------------------------------------------------------------
# Page 1 — Dataset Setup
# ---------------------------------------------------------------------------
def page_dataset_setup() -> None:
    st.header("1 — Dataset Setup")
    st.caption("Load TCGA RNA-seq data, run quality checks, drop uninformative genes, generate EDA figures.")

    src = st.radio("Data source", ["Kaggle (auto-download)", "Local folder"], horizontal=True)
    local_dir = None
    if src == "Local folder":
        local_dir = st.text_input(
            "Path to folder containing data.csv and labels.csv",
            value=str(ROOT / "data"))

    if ARTEFACTS["cleaned_features"].exists():
        st.info("Cleaned dataset already on disk — re-run only to refresh it.")

    if st.button("▶ Run Data Cleaning & EDA", type="primary"):
        bar = st.progress(0.0, text="Starting ...")
        try:
            summary = pipeline.run_data_cleaning_eda(local_dir, progress=_progress_callback(bar))
            st.session_state["eda_summary"] = summary
            st.success("Section 1 complete.")
        except Exception as exc:
            st.error(f"Failed: {exc}")
            return

    summary = st.session_state.get("eda_summary")
    if summary is None and ARTEFACTS["eda_summary"].exists():
        summary = json.loads(ARTEFACTS["eda_summary"].read_text())

    if summary:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Samples", summary["cleaned_shape"].split("x")[0].strip().split()[0])
        c2.metric("Genes (before)", summary["original_shape"].split("x")[1].strip().split()[0])
        c3.metric("Genes (after)", summary["cleaned_shape"].split("x")[1].strip().split()[0])
        c4.metric("Sparsity", summary["matrix_sparsity"])
        c5.metric("Genes removed", summary["total_genes_removed"])

        st.markdown("### EDA figures")
        for fname, title in [
            ("eda_outputs/01_class_distribution.png", "Class distribution"),
            ("eda_outputs/02_sparsity_analysis.png",  "Sparsity analysis"),
            ("eda_outputs/03_top_variance_genes.png", "Top-variance genes"),
            ("eda_outputs/04_correlation_heatmap.png", "Correlation heatmap"),
        ]:
            p = ROOT / fname
            if p.exists():
                st.markdown(f"**{title}**")
                st.image(str(p), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Feature Engineering
# ---------------------------------------------------------------------------
def page_feature_engineering() -> None:
    st.header("2 — Feature Engineering")
    st.caption("ANOVA F-tests, BH-FDR correction, eta-squared, Kruskal-Wallis validation, top-K subsets, PCA-95%.")

    if not ARTEFACTS["cleaned_features"].exists():
        st.warning("Cleaned features not found. Run Page 1 first.")
        return

    if ARTEFACTS["features_pca95"].exists():
        st.info("Feature artefacts already on disk — re-run only to refresh.")

    if st.button("▶ Run Feature Engineering", type="primary"):
        bar = st.progress(0.0, text="Starting ...")
        try:
            summary = pipeline.run_feature_engineering(progress=_progress_callback(bar))
            st.session_state["feature_summary"] = summary
            st.success("Section 2 complete.")
        except Exception as exc:
            st.error(f"Failed: {exc}")
            return

    summary = st.session_state.get("feature_summary")
    if summary is None and ARTEFACTS["feature_summary"].exists():
        summary = json.loads(ARTEFACTS["feature_summary"].read_text())

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Significant genes (FDR<0.05)", f"{summary['significant_genes_fdr_005']:,}")
        c2.metric("Large-effect genes (η²≥0.14)", f"{summary['large_effect_genes_eta_014']:,}")
        c3.metric("PCA components (95%)", summary["pca_components_95var"])
        c4.metric("Spearman ρ (ANOVA vs KW)", f"{summary['spearman_anova_kw']:.3f}")

        for fname, title in [
            ("feature_outputs/01_anova_volcano.png", "Volcano (effect size vs −log10 FDR)"),
            ("feature_outputs/01b_anova_vs_kw.png",  "ANOVA vs Kruskal-Wallis comparison"),
            ("feature_outputs/02_pca_variance.png",  "PCA cumulative variance"),
            ("feature_outputs/03_pca_2d.png",        "PCA 2D scatter by tumor class"),
            ("feature_outputs/04_pca_3d.png",        "PCA 3D scatter by tumor class"),
        ]:
            p = ROOT / fname
            if p.exists():
                st.markdown(f"**{title}**")
                st.image(str(p), use_container_width=True)

        if ARTEFACTS["anova_stats"].exists():
            st.markdown("### Top 20 most discriminative genes")
            top = pd.read_csv(ARTEFACTS["anova_stats"]).head(20)
            st.dataframe(top, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3 — Model Training
# ---------------------------------------------------------------------------
def page_model_training() -> None:
    st.header("3 — Model Training")
    st.caption("Stratified 80/20 split → StandardScaler on train only → 5-fold CV → 8 classifiers.")

    if not ARTEFACTS["features_pca95"].exists():
        st.warning("PCA features not found. Run Page 2 first.")
        return

    feature_choice = st.selectbox(
        "Feature set", ["PCA-95%", "Top-20", "Top-50", "Top-100", "Top-500"])
    all_models = list(make_models().keys())
    chosen_models = st.multiselect("Models to train", all_models, default=all_models)
    cv_folds = st.select_slider("CV folds", options=[3, 5, 10], value=5)

    if st.button("▶ Train Selected Models", type="primary"):
        bar = st.progress(0.0, text="Starting ...")
        try:
            df_res, state = pipeline.run_model_training(
                feature_choice, chosen_models, cv_folds,
                progress=_progress_callback(bar))
            st.session_state["train_state"] = state
            st.success(f"Trained {len(chosen_models)} models — best: {df_res.iloc[0]['Model']}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    df_res = None
    if st.session_state.get("train_state"):
        df_res = st.session_state["train_state"]["results"]
    elif ARTEFACTS["model_results"].exists():
        df_res = pd.read_csv(ARTEFACTS["model_results"])

    if df_res is not None and len(df_res):
        st.markdown("### Results")
        st.dataframe(df_res.round(4), use_container_width=True)

        metric = st.selectbox("Bar chart metric",
                              ["CV_Accuracy_Mean", "Test_Accuracy", "Test_Precision",
                               "Test_Recall", "Test_F1"])
        st.plotly_chart(
            px.bar(df_res.sort_values(metric), x=metric, y="Model",
                   orientation="h", color=metric, color_continuous_scale="Viridis",
                   title=f"Models ranked by {metric}"),
            use_container_width=True)

        st.plotly_chart(
            px.scatter(df_res, x="Training_Time", y="CV_Accuracy_Mean",
                       text="Model", title="Accuracy vs training time"),
            use_container_width=True)

        st.success(f"Best model on CV accuracy: **{df_res.iloc[0]['Model']}**")


# ---------------------------------------------------------------------------
# Page 4 — Best Model & Tuning
# ---------------------------------------------------------------------------
def page_tuning() -> None:
    st.header("4 — Best Model & Tuning")
    state = st.session_state.get("train_state")
    if state is None:
        st.warning("No trained models in this session. Run Page 3 first.")
        return

    best_name = state["results"].iloc[0]["Model"]
    best_acc = float(state["results"].iloc[0]["Test_Accuracy"])
    st.success(f"Top model: **{best_name}**  (test acc {best_acc:.4f})")

    with st.expander("Hyperparameter grid that will be searched"):
        st.json(PARAM_GRIDS.get(best_name, {}))

    if st.button("▶ Run Hyperparameter Tuning", type="primary"):
        bar = st.progress(0.0, text="Starting ...")
        try:
            tuning_out = pipeline.run_hyperparameter_tuning(
                state, progress=_progress_callback(bar))
            st.session_state["tuning_results"] = tuning_out
        except Exception as exc:
            st.error(f"Tuning failed: {exc}")
            return

    tuning_out = st.session_state.get("tuning_results")
    if tuning_out is None and ARTEFACTS["tuning_results"].exists():
        tuning_out = json.loads(ARTEFACTS["tuning_results"].read_text())

    if tuning_out:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best CV", f"{tuning_out['best_cv_score']:.4f}")
        c2.metric("Test acc (after)",
                  f"{tuning_out['test_accuracy_after_tuning']:.4f}",
                  delta=f"{tuning_out['improvement']:+.4f}")
        c3.metric("Improvement", f"{tuning_out['improvement']*100:+.2f}%")

        st.markdown("### Best hyperparameters")
        st.table(pd.DataFrame({
            "Parameter": list(tuning_out["best_params"].keys()),
            "Value":     [str(v) for v in tuning_out["best_params"].values()],
        }))

        if state.get("tuned_pred") is not None:
            y_test = state["y_test"]
            y_pred = state["tuned_pred"]

            st.markdown("### Classification report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

            st.markdown("### Confusion matrix")
            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels, colorscale="Blues",
                text=cm, texttemplate="%{text}",
                hovertemplate="True %{y} → Predicted %{x}: %{z}<extra></extra>"))
            fig.update_layout(xaxis_title="Predicted", yaxis_title="True",
                              title=f"{tuning_out['best_model']} (tuned)")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5 — Prediction
# ---------------------------------------------------------------------------
def page_prediction() -> None:
    st.header("5 — Prediction")
    state = st.session_state.get("train_state")
    if state is None and not ARTEFACTS["best_model"].exists():
        st.warning("No models available. Run Page 3 (and optionally Page 4) first.")
        return

    available = {}
    if state:
        available.update(state["models"])
        if state.get("tuned_model") is not None:
            available["Tuned best model"] = state["tuned_model"]
    if ARTEFACTS["best_model"].exists() and "Tuned best model" not in available:
        available["Tuned best model (saved)"] = joblib.load(ARTEFACTS["best_model"])

    if not available:
        st.warning("No models loaded.")
        return

    model_name = st.selectbox("Model", list(available.keys()))
    model = available[model_name]

    method = st.radio("Input method",
                      ["Random sample from test set",
                       "Upload CSV",
                       "Manual sliders (top-20 genes)"])

    feat_choice = state["feature_choice"] if state else "PCA-95%"
    X_feat = pd.read_csv(feature_set_path(feat_choice), index_col=0)

    if method == "Random sample from test set":
        if state is None:
            st.info("Random-sample mode requires a session-trained model. Use a different input method or retrain.")
            return
        if st.button("▶ Predict on a random test sample"):
            X_test_s = state["X_test_s"]
            y_test = state["y_test"].reset_index(drop=True)
            idx = np.random.randint(0, len(X_test_s))
            sample = X_test_s[idx:idx+1]
            pred = model.predict(sample)[0]
            true = y_test.iloc[idx]

            c1, c2 = st.columns(2)
            c1.metric("Predicted", pred)
            c2.metric("True label", true,
                      delta="✔ correct" if pred == true else "✘ wrong",
                      delta_color="normal" if pred == true else "inverse")
            _show_proba(model, sample)
            _show_class_info(pred)

    elif method == "Upload CSV":
        st.caption(f"Upload a CSV whose columns match `{feat_choice}` features ({X_feat.shape[1]} columns).")
        up = st.file_uploader("CSV file", type=["csv"])
        if up and st.button("▶ Predict"):
            try:
                df = pd.read_csv(up, index_col=0)
                aligned = df.reindex(columns=X_feat.columns).fillna(X_feat.mean())
                if state:
                    sample = state["scaler"].transform(aligned)
                else:
                    from sklearn.preprocessing import StandardScaler
                    sample = StandardScaler().fit(X_feat).transform(aligned)
                preds = model.predict(sample)
                st.dataframe(pd.DataFrame({"sample": df.index, "prediction": preds}),
                             use_container_width=True)
                if len(sample) == 1:
                    _show_proba(model, sample)
                    _show_class_info(preds[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    else:  # Manual sliders
        top20_path = ROOT / "features_top20.csv"
        if not top20_path.exists() or state is None:
            st.info("Manual mode needs a session-trained model and the top-20 feature CSV.")
            return
        top20_df = pd.read_csv(top20_path, index_col=0)
        st.caption(f"Set values for the {top20_df.shape[1]} most discriminative genes.")
        means = top20_df.mean(); mins = top20_df.min(); maxs = top20_df.max()
        values = {}
        cols = st.columns(2)
        for i, gene in enumerate(top20_df.columns):
            with cols[i % 2]:
                values[gene] = st.slider(gene, float(mins[gene]), float(maxs[gene]),
                                          float(means[gene]))
        if st.button("▶ Predict"):
            full = pd.DataFrame(index=[0], columns=X_feat.columns).fillna(X_feat.mean().to_dict())
            for g, v in values.items():
                if g in full.columns:
                    full.iloc[0, full.columns.get_loc(g)] = v
            sample = state["scaler"].transform(full)
            pred = model.predict(sample)[0]
            st.metric("Predicted", pred)
            _show_proba(model, sample)
            _show_class_info(pred)


def _show_proba(model, sample) -> None:
    if not hasattr(model, "predict_proba"):
        return
    proba = model.predict_proba(sample)[0]
    proba_df = (pd.DataFrame({"Class": model.classes_, "Probability": proba})
                  .sort_values("Probability", ascending=False))
    st.plotly_chart(
        px.bar(proba_df, x="Class", y="Probability", color="Class",
               title="Class probabilities"),
        use_container_width=True)


def _show_class_info(cls: str) -> None:
    desc = CLASS_DESCRIPTIONS.get(cls)
    if desc:
        st.info(f"**{cls}** — {desc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PAGES = {
    "1 — Dataset Setup":        page_dataset_setup,
    "2 — Feature Engineering":  page_feature_engineering,
    "3 — Model Training":       page_model_training,
    "4 — Best Model & Tuning":  page_tuning,
    "5 — Prediction":           page_prediction,
}


def main() -> None:
    st.markdown("# 🧬 Cancer RNA-Seq Classification")
    st.caption("STAT 654 Project — TCGA Pan-Cancer multi-class classifier (BRCA / KIRC / LUAD / COAD / PRAD)")
    st.markdown("---")

    page = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Working directory:\n`{ROOT}`")
    st.sidebar.caption("Pipeline logic lives in `cancer_rnaseq/`.")

    PAGES[page]()


if __name__ == "__main__":
    main()
