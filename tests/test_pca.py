"""
Title: test_pca.py
Author: Owen Sharpe
Description: Tests for the PCA class - fit, predict (transform) shape, score is float, and unfitted error.
"""

import numpy as np
import pytest
from velum.classical.pca import PCA


def _make_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 5))


def test_fit():
    X = _make_data()
    model = PCA(n_components=2)
    result = model.fit(X)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X = _make_data()
    model = PCA(n_components=2).fit(X)
    X_transformed = model.predict(X)
    assert X_transformed.shape == (50, 2)


def test_score():
    X = _make_data()
    model = PCA(n_components=2).fit(X)
    score = model.score(X)
    assert isinstance(score, float)


def test_predict_before_fit():
    model = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((10, 5)))
