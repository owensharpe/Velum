"""
Title: test_autoencoder.py
Author: Owen Sharpe
Description: Tests for the AutoEncoder class - fit, predict shape, encode shape, score, and unfitted error.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from velum.deep.autoencoder import AutoEncoder


def _make_data():
    X, _ = make_classification(n_samples=100, n_features=10, random_state=42)
    return X.astype(np.float32)


def test_fit():
    X = _make_data()
    model = AutoEncoder(encoder_layers=[8], latent_dim=4, epochs=3, batch_size=16)
    result = model.fit(X)
    assert model.is_fitted
    assert result is model


def test_predict_shape():
    X = _make_data()
    model = AutoEncoder(encoder_layers=[8], latent_dim=4, epochs=3, batch_size=16)
    model.fit(X)
    pred = model.predict(X)
    assert pred.shape == X.shape


def test_encode_shape():
    X = _make_data()
    model = AutoEncoder(encoder_layers=[8], latent_dim=4, epochs=3, batch_size=16)
    model.fit(X)
    encoded = model.encode(X)
    assert encoded.shape == (len(X), 4)


def test_score():
    X = _make_data()
    model = AutoEncoder(encoder_layers=[8], latent_dim=4, epochs=3, batch_size=16)
    model.fit(X)
    score = model.score(X)
    assert isinstance(score, float)
    assert score >= 0.0


def test_predict_before_fit():
    model = AutoEncoder()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10)))


def test_encode_before_fit():
    model = AutoEncoder()
    with pytest.raises(RuntimeError):
        model.encode(np.zeros((5, 10)))
