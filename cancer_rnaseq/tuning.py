"""Per-model hyperparameter grids and a thin GridSearchCV runner."""
from __future__ import annotations
from typing import Tuple
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .config import RNG
from .modeling import make_models

# Hyperparameter search grids for all 8 models
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
        "solver": ["lbfgs", "liblinear"],  # Optimization algorithm
        "max_iter": [2000],  # Maximum iterations
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],  # Number of trees
        "max_depth": [10, 20, None],  # Tree depth (None = unlimited)
        "min_samples_split": [2, 5],  # Min samples to split a node
    },
    "SVM (RBF)": {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
        "gamma": ["scale", "auto", 0.01, 0.1],  # Kernel coefficient
    },
    "SVM (Linear)": {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 9, 15],  # Number of neighbors
        "weights": ["uniform", "distance"],  # Weight function
        "metric": ["euclidean", "manhattan"],  # Distance metric
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],  # Number of boosting stages
        "learning_rate": [0.05, 0.1, 0.2],  # Shrinkage parameter
        "max_depth": [3, 5],  # Tree depth
    },
    "Decision Tree": {
        "max_depth": [5, 10, 20, None],  # Tree depth (None = unlimited)
        "min_samples_split": [2, 5, 10],  # Min samples to split
        "criterion": ["gini", "entropy"],  # Split quality measure
    },
    "Naive Bayes": {
        "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],  # Variance smoothing
    },
}


def grid_for(model_name: str) -> dict:
    """Get hyperparameter grid for a specific model."""
    return PARAM_GRIDS[model_name]


def tune(model_name: str, X_train_s, y_train, X_test_s, y_test,
         cv_folds: int = 5) -> Tuple[GridSearchCV, float]:
    """Run GridSearchCV on the named model and return (search, test_accuracy)."""
    # Get base model with default parameters
    base = make_models()[model_name]
    
    # Get hyperparameter grid for this model
    grid = PARAM_GRIDS[model_name]
    
    # Set up stratified cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RNG)
    
    # Run grid search with cross-validation
    gs = GridSearchCV(base, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train_s, y_train)
    
    # Evaluate best model on test set
    test_acc = accuracy_score(y_test, gs.predict(X_test_s))
    
    return gs, float(test_acc)
