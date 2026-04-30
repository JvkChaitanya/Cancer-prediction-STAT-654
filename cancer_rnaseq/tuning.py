"""Per-model hyperparameter grids and a thin GridSearchCV runner."""
from __future__ import annotations

from typing import Tuple

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import RNG
from .modeling import make_models


PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [2000],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    },
    "SVM (RBF)": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
    },
    "SVM (Linear)": {
        "C": [0.1, 1, 10, 100],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 9, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "Decision Tree": {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],
    },
}


def grid_for(model_name: str) -> dict:
    return PARAM_GRIDS[model_name]


def tune(model_name: str, X_train_s, y_train, X_test_s, y_test,
         cv_folds: int = 5) -> Tuple[GridSearchCV, float]:
    """Run GridSearchCV on the named model and return (search, test_accuracy)."""
    base = make_models()[model_name]
    grid = PARAM_GRIDS[model_name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RNG)
    gs = GridSearchCV(base, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train_s, y_train)
    test_acc = accuracy_score(y_test, gs.predict(X_test_s))
    return gs, float(test_acc)
