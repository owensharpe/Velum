"""
Title: decision_tree.py
Author: Owen Sharpe
Description: A sklearn-wrapped class representation of a standard Decision Tree Classifier.
"""

from velum.base import BaseModel
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(BaseModel):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        super().__init__()
        self.model_name = "Decision Tree"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self._model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
        )

    def fit(self, X, y):
        """Fit the Decision Tree to training data.

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
