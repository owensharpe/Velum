"""
Title: knn.py
Author: Owen Sharpe
Description: A sklearn-wrapped class representation of a K-Nearest Neighbors Classifier.
"""

from velum.base import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class KNN(BaseModel):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.model_name = "K-Nearest Neighbors"
        self.n_neighbors = n_neighbors
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        """Fit the KNN classifier to training data.

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
