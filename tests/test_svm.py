"""
Title: test_svm.py
Author: Owen Sharpe
Description: Tests for the SVM class - fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from velum.classical.svm import SVM


def _make_data():
    X = np.array([[1, 0], [2, 1], [1.5, 0.5], [2.5, 1.5],
                  [5, 4], [6, 5], [5.5, 4.5], [6.5, 5.5]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def test_fit():
    X, y = _make_data()
    model = SVM()
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X, y = _make_data()
    model = SVM().fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score():
    X, y = _make_data()
    model = SVM().fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_before_fit():
    model = SVM()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 0]]))
