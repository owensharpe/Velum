"""
Title: test_kmeans.py
Author: Owen Sharpe
Description: Tests for the KMeans class - fit, predict shape, score is float, and unfitted error.
"""

import numpy as np
import pytest
from velum.classical.kmeans import KMeans


def _make_data():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(loc=[0, 0], scale=0.5, size=(20, 2)),
                   rng.normal(loc=[5, 5], scale=0.5, size=(20, 2))])
    return X


def test_fit():
    X = _make_data()
    model = KMeans(n_clusters=2, random_state=42)
    result = model.fit(X)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X = _make_data()
    model = KMeans(n_clusters=2, random_state=42).fit(X)
    labels = model.predict(X)
    assert labels.shape == (len(X),)


def test_score():
    X = _make_data()
    model = KMeans(n_clusters=2, random_state=42).fit(X)
    score = model.score(X)
    assert isinstance(score, float)


def test_predict_before_fit():
    model = KMeans()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, 2]]))
