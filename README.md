# Cancer RNA-Seq Classification — STAT_PRO_2

A multi-class classifier that identifies the cancer type of a patient's tumour from RNA-seq gene-expression profiles. Uses the TCGA Pan-Cancer subset (801 patients × 20,531 genes) and distinguishes between **BRCA**, **KIRC**, **LUAD**, **COAD**, and **PRAD**.

The project ships as **two interchangeable interfaces** that share the same pipeline logic and the same on-disk artefacts:

| Component | File | Purpose |
|---|---|---|
| Documented notebook | [`Cancer_RNASeq_Classification.ipynb`](Cancer_RNASeq_Classification.ipynb) | Self-contained walkthrough — every step's code, math, and explanation inline so you can read it top-to-bottom. Intended for review and course submission. |
| Pipeline package | [`cancer_rnaseq/`](cancer_rnaseq/) | Modular Python package implementing the same pipeline as a real software project (data loading, cleaning, EDA, feature selection, PCA, modeling, tuning, orchestration). |
| Interactive Streamlit app | [`app.py`](app.py) | Thin 5-page web UI on top of `cancer_rnaseq` — handles widgets and display, no ML logic of its own. |

> The notebook and the package are **two parallel implementations of the same pipeline**, kept in sync. Pick the notebook if you want to read the project end-to-end; pick the app if you want to run it interactively or extend it cleanly.

---

## What the pipeline does

```
Raw TCGA data (801 × 20,531)
        │
        ▼
SECTION 1 — Cleaning & EDA
   • drop constant / near-constant genes
   • quality checks, sparsity, class balance
   • 4 EDA figures
        │  → cleaned_features.csv, cleaned_labels.csv, eda_summary.json
        ▼
SECTION 2 — Feature Engineering
   • ANOVA F-test per gene + Benjamini-Hochberg FDR
   • eta-squared effect sizes
   • Kruskal-Wallis non-parametric validation (Spearman ρ + top-K overlap)
   • Top-{20, 50, 100, 500} subsets
   • PCA on all genes → 95 % variance (~529 components)
        │  → features_pca95.csv, features_top*.csv, scaler.pkl, pca_model.pkl
        ▼
SECTION 3 — Model Training
   • stratified 80/20 split → StandardScaler fitted on TRAIN only
   • 8 classifiers compared with 5-fold stratified CV
       Logistic Regression, Random Forest, SVM (RBF & Linear),
       KNN, Gradient Boosting, Decision Tree, Naive Bayes
   • GridSearchCV on the winning model
        │  → model_comparison_results.csv,
              hyperparameter_tuning_results.json,
              best_model_tuned.pkl
        ▼
        Predictions on new samples (Streamlit page 5)
```

Reproducibility: every random operation uses `RNG = 42`.

---

## Repository layout

```
STAT_PRO_2/
├── README.md                            ← this file
├── requirements.txt                     ← Python dependencies
├── run_pipeline.sh                      ← headless notebook runner
│
├── Cancer_RNASeq_Classification.ipynb   ← THE NOTEBOOK (3 sections, self-contained)
├── app.py                               ← THE STREAMLIT APP (thin UI, ~340 lines)
│
├── cancer_rnaseq/                       ← THE PIPELINE PACKAGE (used by app.py)
│   ├── __init__.py
│   ├── config.py                        ← constants & artefact paths
│   ├── data_loading.py                  ← Kaggle / local raw loaders
│   ├── data_cleaning.py                 ← QC + low-variance filter
│   ├── eda.py                           ← EDA figures + summary
│   ├── feature_selection.py             ← ANOVA, FDR, η², KW, top-K
│   ├── pca_pipeline.py                  ← scaler + PCA fit/save
│   ├── visualization.py                 ← matplotlib helpers
│   ├── modeling.py                      ← model factory, split, train loop
│   ├── tuning.py                        ← param grids + GridSearchCV
│   └── pipeline.py                      ← high-level orchestrators
│
├── eda_outputs/        ← created by Section 1 / Page 1
├── feature_outputs/    ← created by Section 2 / Page 2
├── model_outputs/      ← created by Section 3 / Pages 3–4
│
├── cleaned_features.csv, cleaned_labels.csv, eda_summary.json
├── features_top{20,50,100,500}.csv, features_pca95.csv
├── anova_feature_stats.csv, kw_feature_stats.csv
├── feature_engineering_summary.json, selected_feature_sets.json
├── scaler.pkl, pca_model.pkl
├── model_comparison_results.csv, hyperparameter_tuning_results.json
└── best_model_tuned.pkl
```

The notebook is monolithic on purpose — every step's code is inline so a reader can scroll through the full implementation. The Streamlit app is thin — it imports orchestrators from `cancer_rnaseq.pipeline` and just renders widgets. Both produce the same artefacts.

The CSV / JSON / PKL files are created when you run the pipeline; they don't ship with the repo.

---

## Prerequisites

- **Python 3.8+**
- ~2 GB free disk space (raw + cleaned data + PCA matrix)
- A few minutes of CPU time (Gradient Boosting CV is the slowest step)
- **Optional**: Kaggle API credentials for auto-download. Place `kaggle.json` at:
  - macOS / Linux: `~/.kaggle/kaggle.json`
  - Windows: `%USERPROFILE%\.kaggle\kaggle.json`

  If you don't have Kaggle access, download `data.csv` and `labels.csv` manually from the [UCI Gene Expression Cancer RNA-Seq dataset](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) and drop both files into a `data/` folder inside `STAT_PRO_2/`.

---

## Step-by-step setup

### 1 — Clone or download the project
```bash
cd STAT_PRO_2
```

### 2 — Create a virtual environment (recommended)

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the notebook

The notebook is self-contained: it loads the data, runs all three pipeline sections, and writes every artefact the Streamlit app expects.

### Option A — Interactive (Jupyter)
```bash
jupyter notebook Cancer_RNASeq_Classification.ipynb
```
Then **Kernel → Restart & Run All** (or step through cell-by-cell). Total runtime ≈ 5–15 minutes depending on hardware (Gradient Boosting CV dominates).

### Option B — VS Code
Open `Cancer_RNASeq_Classification.ipynb` in VS Code, pick a Python interpreter that has the `requirements.txt` dependencies, then click **Run All**.

### Option C — Headless / one-shot
```bash
bash run_pipeline.sh
```
Or directly:
```bash
jupyter nbconvert --to notebook --execute --inplace Cancer_RNASeq_Classification.ipynb
```

What to look for at the end:
- `cleaned_features.csv`, `cleaned_labels.csv`, `features_pca95.csv`, `best_model_tuned.pkl` all exist
- `eda_outputs/`, `feature_outputs/`, `model_outputs/` each contain their figures
- `hyperparameter_tuning_results.json` lists the best model and tuned test accuracy

---

## Running the Streamlit app

```bash
streamlit run app.py
```

This opens the dashboard at **http://localhost:8501**. Use the sidebar to navigate the five pages **in order** the first time — each page guards against missing prerequisites, so you can't skip ahead before the artefacts exist.

### Page 1 — Dataset Setup
1. Pick a data source: **Kaggle (auto-download)** or **Local folder** (point to a folder containing `data.csv` and `labels.csv`).
2. Click **▶ Run Data Cleaning & EDA**.
3. After completion you'll see metrics (samples / genes before & after / sparsity / classes), a class-distribution chart, and the four EDA figures.

> Skip this page if you've already executed the notebook — `cleaned_features.csv` and friends are reused automatically.

### Page 2 — Feature Engineering
1. Click **▶ Run Feature Engineering** (this is the slowest stage — ANOVA + Kruskal-Wallis across ~20,000 genes).
2. Watch the progress bar; expect ~2–4 minutes.
3. View: significance / effect-size metrics, the volcano plot, the ANOVA-vs-KW comparison, PCA cumulative variance, the PCA 2D scatter, and a top-20 gene table.

### Page 3 — Model Training
1. Choose a **feature set** (`PCA-95%`, `Top-20`, `Top-50`, `Top-100`, `Top-500`).
2. Pick the **models** you want trained (all 8 are pre-selected).
3. Choose **CV folds** (3 / 5 / 10 — default 5).
4. Click **▶ Train Selected Models**.
5. Inspect the results table, the interactive metric bar chart, and the accuracy-vs-time scatter. The best model is announced at the bottom.

### Page 4 — Best Model & Tuning
1. Confirms the top model from Page 3.
2. Click **▶ Run Hyperparameter Tuning** to launch GridSearchCV with the appropriate grid.
3. After completion: before-vs-after accuracy delta, the chosen hyperparameters, the per-class classification report, and an interactive Plotly confusion matrix.

### Page 5 — Prediction
Pick a model and an input method:
- **Random sample from test set** — predicts on a randomly chosen held-out sample and shows whether the prediction matches the true label.
- **Upload CSV** — upload a CSV with the same gene/PC columns as the active feature set; results show predictions and (when one row) probabilities.
- **Manual sliders (top-20 genes)** — interactively adjust expression values for the 20 most discriminative genes.

The probability bar chart and a one-line description of the predicted cancer type appear under every prediction.

---

## Expected results (from STAT_PRO baseline)

| Model | CV Accuracy | Test Accuracy | Test F1 |
|---|---|---|---|
| **Gradient Boosting** | **0.968 ± 0.027** | **0.945** | **0.945** |
| Decision Tree | 0.907 ± 0.022 | 0.866 | 0.863 |
| Random Forest | 0.867 ± 0.029 | 0.811 | 0.801 |
| SVM (Linear) | 0.790 ± 0.036 | 0.796 | 0.783 |
| Logistic Regression | 0.797 ± 0.043 | 0.791 | 0.780 |
| Naive Bayes | 0.682 ± 0.065 | 0.652 | 0.655 |
| SVM (RBF) | 0.548 ± 0.013 | 0.537 | 0.454 |
| KNN (k=5) | 0.265 ± 0.026 | 0.244 | 0.176 |

Gradient Boosting consistently wins; KNN collapses (curse of dimensionality on ~529 PCA components).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `kagglehub` fails on Page 1 | Switch to **Local folder** mode and provide `data.csv` + `labels.csv` manually, or set up `kaggle.json` credentials. |
| Page 2 hangs for several minutes | Normal — ~20,000 ANOVA + ~20,000 Kruskal-Wallis tests. Watch the progress bar. |
| Page 3 says *"PCA features not found"* | Run Page 2 first (or execute the notebook end-to-end). |
| Page 4 says *"No trained models in this session"* | Page 4 needs models trained **in the current Streamlit session** (in-memory state). Re-run Page 3 in the same browser session. |
| `streamlit: command not found` | Activate your virtualenv, then `pip install -r requirements.txt`. On Windows you may need `python -m streamlit run app.py`. |
| Out-of-memory during PCA | Close other processes; the standardised matrix is 801 × 20,221 floats (~130 MB). |

---

## License & attribution

- **Dataset**: UCI Machine Learning Repository — *Gene Expression Cancer RNA-Seq* (TCGA Pan-Cancer subset). See Weinstein et al. (2013), *Nature Genetics* 45, 1113–1120.
- **Course**: STAT 654 — Machine Learning, 2026.
