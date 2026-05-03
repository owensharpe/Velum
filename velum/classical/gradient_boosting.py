"""
Title: gradient_boosting.py
Author: Owen Sharpe
Description: A sklearn-wrapped class representation of a Gradient Boosting Classifier.
"""

from velum.base import BaseModel
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoosting(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        super().__init__()
        self.model_name = "Gradient Boosting"
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Fit the Gradient Boosting classifier to training data.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target class labels, shape (n_samples,)

        Returns:
            self: The fitted model.
        """
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict class labels for the given input features.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            predictions: Predicted class labels, shape (n_samples,)
        """
        self._check_is_fitted()
        return self._model.predict(X)

    def score(self, X, y):
        """Compute mean accuracy on the given data.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            y: True class labels, shape (n_samples,)

        Returns:
            score: Mean accuracy as a float in [0, 1].
        """
        self._check_is_fitted()
        return self._model.score(X, y)
