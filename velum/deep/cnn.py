"""
Title: cnn.py
Author: Owen Sharpe
Description: CNN model supporting 1D convolutions (sequences) and 2D convolutions (images).
"""

import torch.nn as nn
from velum.deep.base_deep import BaseDeepModel


class _CNN1DNetwork(nn.Module):
    def __init__(self, channels, kernel_size, pooling, output_size):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(2) if pooling == 'max' else nn.AvgPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], output_size)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.pool(self.conv(x)).squeeze(-1)  # (batch, channels[-1])
        return self.fc(x)


class _CNN2DNetwork(nn.Module):
    def __init__(self, channels, kernel_size, pooling, output_size):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool2d(2) if pooling == 'max' else nn.AvgPool2d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], output_size)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (batch, 1, H, W)
        x = self.pool(self.conv(x)).flatten(1)  # (batch, channels[-1])
        return self.fc(x)


class CNN(BaseDeepModel):
    def __init__(
        self,
        task='classification',
        channels=None,
        kernel_size=3,
        pooling='max',
        input_dims=1,
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
        self.model_name = "CNN"
        self.channels = channels or [32, 64]
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.input_dims = input_dims

    def _build_network(self):
        """Build 1D or 2D convolutional network based on input_dims."""
        if self.input_dims == 1:
            return _CNN1DNetwork(self.channels, self.kernel_size, self.pooling, self.output_size)
        return _CNN2DNetwork(self.channels, self.kernel_size, self.pooling, self.output_size)
