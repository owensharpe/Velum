"""
Title: logistic_regression.py
Author: Owen Sharpe
Description: A functional class representation of a standard Logistic Regression.
"""

# import classes
from velum.base import BaseModel

# import libraries
import numpy as np


class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__()
        self.model_name = "Logistic Regression"

        # instantiate function variables
        self.weights = None
        self.bias = None

        # instantiate hyperparameters
        self.lr = learning_rate
        self.n_iterations = n_iterations

    def _sigmoid(self, z):
        """Performs the sigmoid calculation on a given value z.

        Args:
            z: logit input

        Returns:
            sigmoid_val: The sigmoid value.
        """

        # calculate the sigmoid value
        sigmoid_val = 1 / (1 + np.exp(-z))

        return sigmoid_val

    def fit(self, X, y):
        """Fitting a Logistic Regression model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values, shape (n_samples)

        Returns:
            self: The fitted Logistic Regression model.
        """

        # refactor data to Numpy arrays
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y)

        # initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # perform gradient descent
        for i in range(self.n_iterations):

            # compute logits
            z = X @ self.weights + self.bias

            # get probabilities
            pred_y = self._sigmoid(z)

            # compute error
            residuals = pred_y - y

            # compute gradients
            weight_grad = (1 / X.shape[0]) * X.T @ residuals
            bias_grad = (1 / X.shape[0]) * np.sum(residuals)

            self.weights -= weight_grad * self.lr
            self.bias -= bias_grad * self.lr

        # set fitted boolean to true
        self.is_fitted = True

        return self

    def predict_proba(self, X):
        """Predicting the probabilities for the testing values for a Logistic Regression model.

        Args:
            X: Testing features, shape (n_samples, n_features)

        Returns:
            pred_prob: The predicted probabilities, shape (n_samples)
        """

        # check if the model is fitted
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")

        # refactor data
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # compute logits
        z = X @ self.weights + self.bias

        # predict probabilities
        pred_prob = self._sigmoid(z)

        return pred_prob

    def predict(self, X):
        """Predicting the target values for a Logistic Regression model.

        Args:
            X: Testing features, shape (n_samples, n_features)

        Returns:
            pred_y: The predicted target values, shape (n_samples)
        """

        # predict target values
        pred_prob = self.predict_proba(X)
        pred_y = (pred_prob >= 0.5).astype(int)

        return pred_y

    def score(self, X, y):
        """Score the predicted y-values over the target values for a Logistic Regression Model.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            y: True target values, shape (n_samples)

        Returns:
            score: accuracy
        """

        # refactor data
        y = np.array(y)

        # predict target values
        pred_y = self.predict(X)

        # calculate accuracy
        accuracy = np.sum(pred_y == y) / y.shape[0]

        return accuracy
