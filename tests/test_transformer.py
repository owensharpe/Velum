"""
Title: test_transformer.py
Author: Owen Sharpe
Description: Tests for the Transformer class - fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from velum.deep.transformer import Transformer


def _make_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=100)
    return X, y


def test_fit():
    X, y = _make_data()
    model = Transformer(
        task='classification', d_model=8, nhead=2,
        num_encoder_layers=1, dim_feedforward=16, dropout=0.0,
        epochs=3, batch_size=16,
    )
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X, y = _make_data()
    model = Transformer(
        task='classification', d_model=8, nhead=2,
        num_encoder_layers=1, dim_feedforward=16, dropout=0.0,
        epochs=3, batch_size=16,
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score():
    X, y = _make_data()
    model = Transformer(
        task='classification', d_model=8, nhead=2,
        num_encoder_layers=1, dim_feedforward=16, dropout=0.0,
        epochs=3, batch_size=16,
    )
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_before_fit():
    model = Transformer(task='classification')
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10, 8)))
