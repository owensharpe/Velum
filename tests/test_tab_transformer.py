"""
Title: test_tab_transformer.py
Author: Owen Sharpe
Description: Tests for the TabTransformer class - fit, predict shape, score range, and unfitted error.
"""

import numpy as np
import pytest
from velum.deep.tab_transformer import TabTransformer


def _make_data():
    rng = np.random.default_rng(42)
    cat = rng.integers(0, 3, size=(100, 2)).astype(np.float32)  # 2 categorical features, 3 categories each
    cont = rng.standard_normal((100, 3)).astype(np.float32)     # 3 continuous features
    X = np.hstack([cat, cont])
    y = rng.integers(0, 2, size=100)
    return X, y


def test_fit():
    X, y = _make_data()
    model = TabTransformer(
        task='classification', num_categories=[3, 3], num_continuous=3,
        d_model=8, nhead=2, num_layers=1, dropout=0.0,
        epochs=3, batch_size=16,
    )
    result = model.fit(X, y)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X, y = _make_data()
    model = TabTransformer(
        task='classification', num_categories=[3, 3], num_continuous=3,
        d_model=8, nhead=2, num_layers=1, dropout=0.0,
        epochs=3, batch_size=16,
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_score():
    X, y = _make_data()
    model = TabTransformer(
        task='classification', num_categories=[3, 3], num_continuous=3,
        d_model=8, nhead=2, num_layers=1, dropout=0.0,
        epochs=3, batch_size=16,
    )
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_before_fit():
    model = TabTransformer(task='classification', num_categories=[3, 3], num_continuous=3)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 5)))
