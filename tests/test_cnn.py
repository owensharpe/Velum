"""
Title: test_cnn.py
Author: Owen Sharpe
Description: Tests for the CNN class - 1D and 2D, fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from velum.deep.cnn import CNN


def _make_1d_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 20)).astype(np.float32)
    y = rng.integers(0, 2, size=100)
    return X, y


def _make_2d_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 8, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=100)
    return X, y


def test_fit_1d():
    X, y = _make_1d_data()
    model = CNN(task='classification', channels=[8, 16], kernel_size=3, input_dims=1, epochs=3, batch_size=16)
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape_1d():
    X, y = _make_1d_data()
    model = CNN(task='classification', channels=[8, 16], kernel_size=3, input_dims=1, epochs=3, batch_size=16)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score_1d():
    X, y = _make_1d_data()
    model = CNN(task='classification', channels=[8, 16], kernel_size=3, input_dims=1, epochs=3, batch_size=16)
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_fit_2d():
    X, y = _make_2d_data()
    model = CNN(task='classification', channels=[4, 8], kernel_size=3, input_dims=2, epochs=3, batch_size=16)
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape_2d():
    X, y = _make_2d_data()
    model = CNN(task='classification', channels=[4, 8], kernel_size=3, input_dims=2, epochs=3, batch_size=16)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score_2d():
    X, y = _make_2d_data()
    model = CNN(task='classification', channels=[4, 8], kernel_size=3, input_dims=2, epochs=3, batch_size=16)
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_before_fit():
    model = CNN(task='classification')
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 20)))
