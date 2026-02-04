"""
Title: test_logistic_regression.py
Author: Owen Sharpe
Description: Setting up a test to ensure the LogisticRegression class works.
"""

# import classes
from velum.classical.logistic_regression import LogisticRegression

# import libraries
import numpy as np
import pytest


def test_logistic_regression_fit_and_predict():

    # example test

    # class 0
    X_0 = [[1], [2], [1.5], [2.5]]
    y_0 = [0, 0, 0, 0]

    # class 1
    X_1 = [[5], [6], [5.5], [6.5]]
    y_1 = [1, 1, 1, 1]

    # combine
    X = X_0 + X_1
    y = y_0 + y_1

    # fit model
    model = LogisticRegression()
    model.fit(X, y)

    # check that model is fitted
    assert model.is_fitted

    # predictions should be almost identical to original values
    pred_y = model.predict(X)
    assert list(pred_y) == y


def test_logistic_regression_score():

    # example test

    # class 0
    X_0 = [[1], [2], [1.5], [2.5]]
    y_0 = [0, 0, 0, 0]

    # class 1
    X_1 = [[5], [6], [5.5], [6.5]]
    y_1 = [1, 1, 1, 1]

    # combine
    X = X_0 + X_1
    y = y_0 + y_1

    # fit model
    model = LogisticRegression()
    model.fit(X, y)

    # assess score
    score = model.score(X, y)
    assert score > 0.99


def test_logistic_regression_predict_proba_bounds():

    # example test

    # class 0
    X_0 = [[1], [2], [1.5], [2.5]]
    y_0 = [0, 0, 0, 0]

    # class 1
    X_1 = [[5], [6], [5.5], [6.5]]
    y_1 = [1, 1, 1, 1]

    # combine
    X = X_0 + X_1
    y = y_0 + y_1

    # fit model
    model = LogisticRegression()
    model.fit(X, y)

    # assess probability predictions
    pred_prob = model.predict_proba(X)
    assert np.all(pred_prob >= 0) and np.all(pred_prob <= 1)


def test_logistic_regression_unfitted_error():

    # instantiate model
    model = LogisticRegression()

    # assess unfitting error
    with pytest.raises(RuntimeError):
        model.predict([[1], [2]])
