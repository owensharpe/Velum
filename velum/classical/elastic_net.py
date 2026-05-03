"""
Title: elastic_net.py
Author: Owen Sharpe
Description: A sklearn-wrapped class representation of Elastic Net Regression (L1 + L2 regularization).
"""

from velum.base import BaseModel
from sklearn.linear_model import ElasticNet


class ElasticNetRegression(BaseModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000):
        super().__init__()
        self.model_name = "Elastic Net"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    def fit(self, X, y):
        """Fit the Elastic Net model to training data.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            self: The fitted model.
        """
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict target values for the given input features.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            predictions: Predicted target values, shape (n_samples,)
        """
        self._check_is_fitted()
        return self._model.predict(X)

    def score(self, X, y):
        """Compute R-squared score on the given data.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            y: True target values, shape (n_samples,)

        Returns:
            score: R-squared coefficient of determination.
        """
        self._check_is_fitted()
        return self._model.score(X, y)
