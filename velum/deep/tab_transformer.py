"""
Title: tab_transformer.py
Author: Owen Sharpe
Description: TabTransformer model for tabular data with categorical and continuous features.
"""

import torch
import torch.nn as nn
from velum.deep.base_deep import BaseDeepModel


class _TabTransformerNetwork(nn.Module):
    def __init__(self, num_categories, num_continuous, d_model, nhead, num_layers, dropout, output_size):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(nc, d_model) for nc in num_categories])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        n_cat = len(num_categories)
        mlp_input = n_cat * d_model + num_continuous
        hidden = max(mlp_input // 2, output_size)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x):
        n_cat = len(self.embeddings)
        if n_cat > 0:
            cat_x = x[:, :n_cat].long()
            cont_x = x[:, n_cat:]
            embedded = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
            cat_embedded = torch.stack(embedded, dim=1)     # (batch, n_cat, d_model)
            cat_flat = self.transformer(cat_embedded).flatten(1)  # (batch, n_cat * d_model)
            combined = torch.cat([cat_flat, cont_x], dim=1)
        else:
            combined = x
        return self.mlp(combined)


class TabTransformer(BaseDeepModel):
    def __init__(
        self,
        task='classification',
        num_categories=None,
        num_continuous=0,
        d_model=32,
        nhead=4,
        num_layers=2,
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
        self.model_name = "TabTransformer"
        self.num_categories = num_categories or []
        self.num_continuous = num_continuous
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

    def _build_network(self):
        """Build TabTransformer with categorical embeddings, transformer encoder, and MLP head."""
        return _TabTransformerNetwork(
            num_categories=self.num_categories,
            num_continuous=self.num_continuous,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.output_size,
        )
