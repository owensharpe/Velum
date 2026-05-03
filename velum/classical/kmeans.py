"""
Title: kmeans.py
Author: Owen Sharpe
Description: A sklearn-wrapped class representation of K-Means Clustering.
"""

from velum.base import BaseModel
from sklearn.cluster import KMeans as SklearnKMeans


class KMeans(BaseModel):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        super().__init__()
        self.model_name = "K-Means"
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = SklearnKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        """Fit the K-Means model to the input data.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Ignored - included for API consistency.

        Returns:
            self: The fitted model.
        """
        self._model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict the closest cluster label for each sample.

        Args:
            X: Features to cluster, shape (n_samples, n_features)

        Returns:
            labels: Cluster index for each sample, shape (n_samples,)
        """
        self._check_is_fitted()
        return self._model.predict(X)

    def score(self, X, y=None):
        """Compute the negative inertia (sum of squared distances to nearest cluster center).

        Args:
            X: Features to score, shape (n_samples, n_features)
            y: Ignored - included for API consistency.

        Returns:
            score: Negative inertia (higher is better).
        """
        self._check_is_fitted()
        return self._model.score(X)
