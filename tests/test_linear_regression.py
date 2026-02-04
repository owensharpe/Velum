"""
Title: test_linear_regression.py
Author: Owen Sharpe
Description: Setting up a test to ensure the LinearRegression class works.
"""

# import classes
from velum.classical.linear_regression import LinearRegression

# import libraries
import numpy as np


def test_linear_regression_fit_and_predict():

    # example test: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])

    # fit model
    model = LinearRegression()
    model.fit(X, y)

    # check that model is fitted
    assert model.is_fitted

    # predictions should be almost identical to original values
    pred_y = model.predict(X)
    np.testing.assert_array_almost_equal(pred_y, y)


def test_linear_regression_score():

    # example test: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])

    # fit model
    model = LinearRegression()
    model.fit(X, y)

    # assess score
    score = model.score(X, y)
    assert score > 0.99
