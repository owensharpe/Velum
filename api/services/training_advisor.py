"""
Title: training_advisor.py
Author: Owen Sharpe
Description: Heuristics for recommending a model and default hyperparameters
given a dataset's shape and task type. Used by chat_tools.propose_training
and (eventually) by the no-code frontend to suggest sensible defaults.
"""

from typing import Any

# per-model default hyperparameters. Keyed by class_name from AVAILABLE_MODELS.
# value are conservative; sensible for tabular data without much tuning.
_DEFAULT_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 10,
    },
    "LogisticRegression": {
        "max_iter": 1000,
    },
    "LinearRegression": {},
    "RidgeRegression": {
        "alpha": 1.0,
    },
}


def recommend_model(task_type: str, n_rows: int) -> tuple[str, str]:
    """
    Pick a model class name and return a human-readable rationale.

    task_type: "classification" or "regression"
    n_rows: number of rows in the dataset

    Returns: (class_name, rationale)
    """
    if task_type == "classification":
        if n_rows < 500:
            return (
                "LogisticRegression",
                f"With only {n_rows} rows, a simple linear model is more stable "
                "than tree-based methods, which tend to overfit on small data.",
            )
        return (
            "RandomForest",
            "Random forest is a strong default for tabular classification — "
            "handles mixed feature types well and needs little tuning.",
        )

    if task_type == "regression":
        if n_rows < 500:
            return (
                "LinearRegression",
                f"With only {n_rows} rows, plain linear regression is the most "
                "reliable choice — it captures linear relationships and doesn't "
                "need enough data to estimate a regularization strength.",
            )
        return (
            "RidgeRegression",
            "Ridge regression is a strong default for tabular regression — "
            "it's linear regression with L2 regularization, which keeps "
            "coefficients stable and reduces overfitting without much tuning.",
        )

    raise ValueError(f"Unknown task_type: {task_type}")


def default_hyperparameters(class_name: str) -> dict[str, Any]:
    """Return a copy of the default hyperparameters for a model class."""
    return dict(_DEFAULT_HYPERPARAMETERS.get(class_name, {}))
