"""Model factory, stratified split, scaler fit, training/evaluation loop."""
from __future__ import annotations
import time
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from .config import RNG, TEST_SIZE

# Type alias for progress callback function
ProgressFn = Optional[Callable[[float, str], None]]


def make_models() -> Dict[str, object]:
    """Instantiate all 8 baseline classifiers used in this project."""
    # Create dictionary of all models with default hyperparameters
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RNG),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RNG, n_jobs=-1),
        "SVM (RBF)":           SVC(kernel="rbf",    random_state=RNG, probability=True),
        "SVM (Linear)":        SVC(kernel="linear", random_state=RNG, probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=RNG),
        "Decision Tree":       DecisionTreeClassifier(random_state=RNG),
        "Naive Bayes":         GaussianNB(),
    }


def stratified_split(X, y, test_size: float = TEST_SIZE):
    """Return X_train, X_test, y_train, y_test — stratified, deterministic."""
    # Split data while preserving class proportions (stratified)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RNG)


def fit_scaler(X_train) -> Tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on the training matrix and return it."""
    # Fit scaler on training data only (prevent data leakage)
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(X_train)


def evaluate_one(model, X_train_s, y_train, X_test_s, y_test, cv) -> dict:
    """5-fold CV on train + final fit + test-set metrics for a single model."""
    # Start timing
    t0 = time.time()
    
    # Run 5-fold cross-validation on training data
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    
    # Train model on full training set
    model.fit(X_train_s, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test_s)
    
    # Calculate elapsed time
    elapsed = time.time() - t0
    
    # Return all metrics
    return {
        "CV_Accuracy_Mean": cv_scores.mean(),
        "CV_Accuracy_Std":  cv_scores.std(),
        "Test_Accuracy":    accuracy_score(y_test, y_pred),
        "Test_Precision":   precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Test_Recall":      recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "Test_F1":          f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Training_Time":    elapsed,
    }


def train_all_models(model_dict: Dict[str, object],
                     X_train_s, y_train, X_test_s, y_test,
                     cv_folds: int = 5, progress: ProgressFn = None
                     ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train and evaluate every model in `model_dict`. Returns (results_df, trained)."""
    # Set up stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RNG)
    records, trained = [], {}
    n = max(len(model_dict), 1)
    
    # Train each model
    for i, (name, model) in enumerate(model_dict.items()):
        # Report progress
        if progress:
            progress(i / n, f"Training {name} ...")
        
        # Evaluate model
        rec = evaluate_one(model, X_train_s, y_train, X_test_s, y_test, cv)
        rec["Model"] = name
        records.append(rec)
        
        # Store trained model
        trained[name] = model
    
    if progress:
        progress(1.0, "Done.")
    
    # Create results DataFrame sorted by CV accuracy
    cols = ["Model", "CV_Accuracy_Mean", "CV_Accuracy_Std", "Test_Accuracy",
            "Test_Precision", "Test_Recall", "Test_F1", "Training_Time"]
    df = (pd.DataFrame(records)[cols]
            .sort_values("CV_Accuracy_Mean", ascending=False)
            .reset_index(drop=True))
    
    return df, trained
