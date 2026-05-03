"""
Title: test_ridge_regression.py
Author: Owen Sharpe
Description: Tests for the RidgeRegression class - fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from velum.classical.ridge_regression import RidgeRegression


def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    model = RidgeRegression()
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    model = RidgeRegression().fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    model = RidgeRegression().fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_before_fit():
    model = RidgeRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1]]))
