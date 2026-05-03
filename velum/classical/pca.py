"""
Title: pca.py
Author: Owen Sharpe
Description: A sklearn-wrapped class representation of Principal Component Analysis (PCA).
"""

from velum.base import BaseModel
from sklearn.decomposition import PCA as SklearnPCA


class PCA(BaseModel):
    def __init__(self, n_components=None):
        super().__init__()
        self.model_name = "PCA"
        self.n_components = n_components
        self._model = SklearnPCA(n_components=n_components)

    def fit(self, X, y=None):
        """Fit PCA on the input data.

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
        """Project data into the principal component space.

        Args:
            X: Features to transform, shape (n_samples, n_features)

        Returns:
            X_transformed: Projected data, shape (n_samples, n_components)
        """
        self._check_is_fitted()
        return self._model.transform(X)

    def score(self, X, y=None):
        """Compute the average log-likelihood of the data under the fitted model.

        Args:
            X: Features to score, shape (n_samples, n_features)
            y: Ignored - included for API consistency.

        Returns:
            score: Average log-likelihood.
        """
        self._check_is_fitted()
        return self._model.score(X)
