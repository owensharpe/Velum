"""
Title: lstm.py
Author: Owen Sharpe
Description: LSTM model for sequential and time series data.
"""

import torch
import torch.nn as nn
from velum.deep.base_deep import BaseDeepModel


class _LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, output_size):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * directions, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * directions, batch, hidden_size)
        out = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(out)


class LSTM(BaseDeepModel):
    def __init__(
        self,
        task='classification',
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
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
        self.model_name = "LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

    def _get_input_size(self, X_np):
        # X_np is 3D: (n_samples, seq_len, n_features)
        return X_np.shape[2]

    def _build_network(self):
        """Build LSTM network with a linear output head."""
        return _LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            output_size=self.output_size,
        )
