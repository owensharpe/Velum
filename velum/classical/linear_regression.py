"""
Title: linear_regression.py
Author: Owen Sharpe
Description: A functional class representation of a standard Linear Regression.
"""

# import classes
from velum.base import BaseModel

# import libraries
import numpy as np


class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Linear Regression"

        # instantiate function variables
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fitting a Linear Regression model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values, shape (n_samples)

        Returns:
            self: The fitted Linear Regression model.
        """

        # instantiate Numpy arrays
        X = np.column_stack([np.ones(X.shape[0]), X])
        y = np.array(y)

        # solve for Linear Regression weights & bias
        params = np.linalg.inv(X.T @ X) @ X.T @ y
        self.weights = params[1:]
        self.bias = params[0]

        # set fitted boolean to true
        self.is_fitted = True

        return self

    def predict(self, X):
        """Predicting the target values for a Linear Regression model.

        Args:
            X: Testing features, shape (n_samples, n_features)

        Returns:
            pred_y: The predicted target values, shape (n_samples)
        """

        # check if the model is fitted
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")

        # instantiate Numpy arrays
        X = np.array(X)

        # get predictions
        pred_y = X @ self.weights + self.bias

        return pred_y


    def score(self, X, y):
        """Score the predicted y-values over the target values for a Linear Regression Model.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            y: True target values, shape (n_samples)

        Returns:
            score: R-squared (could also maybe use RMSE/MSE in future)
        """

        # get predictions
        pred_y = self.predict(X)

        # compute R-squared metric
        residual_SoS = np.sum((y - pred_y) ** 2)
        total_SoS = np.sum((y - np.mean(y)) ** 2)
        score = 1 - residual_SoS / total_SoS

        return score
