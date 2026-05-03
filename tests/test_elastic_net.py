"""
Title: test_elastic_net.py
Author: Owen Sharpe
Description: Tests for the ElasticNetRegression class - fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from velum.classical.elastic_net import ElasticNetRegression


def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    model = ElasticNetRegression(alpha=0.01)
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    model = ElasticNetRegression(alpha=0.01).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    model = ElasticNetRegression(alpha=0.01).fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_before_fit():
    model = ElasticNetRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1]]))
