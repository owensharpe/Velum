"""
Title: test_mlp.py
Author: Owen Sharpe
Description: Tests for the MLP class - fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from velum.deep.mlp import MLP


def test_fit_classification():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = MLP(task='classification', hidden_layers=[16, 8], epochs=3, batch_size=16)
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape_classification():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = MLP(task='classification', hidden_layers=[16, 8], epochs=3, batch_size=16)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score_classification():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = MLP(task='classification', hidden_layers=[16, 8], epochs=3, batch_size=16)
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_fit_regression():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    model = MLP(task='regression', hidden_layers=[16, 8], epochs=3, batch_size=16)
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape_regression():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    model = MLP(task='regression', hidden_layers=[16, 8], epochs=3, batch_size=16)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score_regression():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    model = MLP(task='regression', hidden_layers=[16, 8], epochs=3, batch_size=16)
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)


def test_predict_before_fit():
    model = MLP(task='classification')
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10)))
