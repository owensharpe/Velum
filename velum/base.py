"""
Title: base.py
Author: Owen Sharpe
Description: A common base class in which all models inherit from
"""


class BaseModel:
    def __init__(self):
        """Base Model Constructor"""
        self.is_fitted = False
        self.model_name = "Base Model"

    def fit(self, X, y):
        """Train the model on a given dataset.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values, shape (n_samples)

        Returns:
            self: The fitted model
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X):
        """Predict y-values using model on given testing data.

        Args:
            X: Testing features, shape (n_samples, n_features)

        Returns:
            predictions: The predicted target values, shape (n_samples)
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def score(self, X, y):
        """Score the predicted y-values over the target values.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            y: True target values, shape (n_samples)

        Returns:
            score: Model performance metric (e.g., accuracy, R-squared, RMSE)
        """
        raise NotImplementedError("Subclasses must implement score()")








