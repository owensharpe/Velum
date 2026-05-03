"""
Title: mlp.py
Author: Owen Sharpe
Description: Multi-layer perceptron (MLP) for classification and regression.
"""

import torch.nn as nn
from velum.deep.base_deep import BaseDeepModel


class MLP(BaseDeepModel):
    def __init__(
        self,
        task='classification',
        hidden_layers=None,
        dropout=0.0,
        activation='relu',
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
        self.model_name = "MLP"
        self.hidden_layers = hidden_layers or [128, 64]
        self.dropout = dropout
        self.activation = activation

    def _activation_fn(self):
        if self.activation == 'relu':
            return nn.ReLU()
        if self.activation == 'tanh':
            return nn.Tanh()
        if self.activation == 'leaky_relu':
            return nn.LeakyReLU()
        raise ValueError(f"Unknown activation: {self.activation}")

    def _build_network(self):
        """Build a fully connected network from hidden_layers with optional dropout."""
        layers = []
        in_size = self.input_size
        for hidden in self.hidden_layers:
            layers.append(nn.Linear(in_size, hidden))
            layers.append(self._activation_fn())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_size = hidden
        layers.append(nn.Linear(in_size, self.output_size))
        return nn.Sequential(*layers)
