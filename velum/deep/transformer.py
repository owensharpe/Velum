"""
Title: transformer.py
Author: Owen Sharpe
Description: Transformer model using PyTorch's nn.TransformerEncoder for attention-based sequential modeling.
"""

import torch.nn as nn
from velum.deep.base_deep import BaseDeepModel


class _TransformerNetwork(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.transformer(x)         # (batch, seq_len, d_model)
        x = x.mean(dim=1)               # global average over sequence
        return self.fc(x)


class Transformer(BaseDeepModel):
    def __init__(
        self,
        task='classification',
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        epochs=100,
        learning_rate=1e-3,
        batch_size=32,
        optimizer='adam',
        scheduler=None,
        patience=None,
        validation_split=0.1,
    ):
        super().__init__(
            task=task,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=patience,
            validation_split=validation_split,
        )
        self.model_name = "Transformer"
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def _get_input_size(self, X_np):
        # X_np is 3D: (n_samples, seq_len, n_features)
        return X_np.shape[2]

    def _build_network(self):
        """Build TransformerEncoder with input projection and linear output head."""
        return _TransformerNetwork(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=self.output_size,
        )
