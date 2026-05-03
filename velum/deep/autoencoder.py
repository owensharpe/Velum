"""
Title: autoencoder.py
Author: Owen Sharpe
Description: Autoencoder for unsupervised representation learning and reconstruction.
"""

import numpy as np
import torch
import torch.nn as nn
from velum.deep.base_deep import BaseDeepModel


class _AutoEncoderNet(nn.Module):
    def __init__(self, input_size, encoder_layers, latent_dim):
        super().__init__()
        enc = []
        in_size = input_size
        for h in encoder_layers:
            enc += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        enc.append(nn.Linear(in_size, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        in_size = latent_dim
        for h in reversed(encoder_layers):
            dec += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        dec.append(nn.Linear(in_size, input_size))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoEncoder(BaseDeepModel):
    def __init__(
        self,
        encoder_layers=None,
        latent_dim=16,
        epochs=100,
        learning_rate=1e-3,
        batch_size=32,
        optimizer='adam',
        scheduler=None,
        patience=None,
        validation_split=0.1,
    ):
        super().__init__(
            task='reconstruction',
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=patience,
            validation_split=validation_split,
        )
        self.model_name = "AutoEncoder"
        self.encoder_layers = encoder_layers or [64, 32]
        self.latent_dim = latent_dim

    def fit(self, X, y=None):
        """Fit the autoencoder to reconstruct the input data.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Ignored - included for API consistency.

        Returns:
            self: The fitted model.
        """
        return super().fit(X, X)

    def _build_network(self):
        """Build the encoder-decoder architecture."""
        return _AutoEncoderNet(self.input_size, self.encoder_layers, self.latent_dim)

    def encode(self, X):
        """Return the latent representation of the input data.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Z: Latent representation, shape (n_samples, latent_dim).
        """
        self._check_is_fitted()
        X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32).to(self.device)
        self.network.eval()
        with torch.no_grad():
            return self.network.encoder(X_t).cpu().numpy()
