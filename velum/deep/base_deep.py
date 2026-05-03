"""
Title: base_deep.py
Author: Owen Sharpe
Description: Base class for all deep learning models in Velum, owning the training loop,
optimizer/scheduler selection, and early stopping.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from velum.base import BaseModel


class BaseDeepModel(BaseModel):
    def __init__(
        self,
        task='classification',
        epochs=100,
        learning_rate=1e-3,
        batch_size=32,
        optimizer='adam',
        scheduler=None,
        patience=None,
        validation_split=0.1,
    ):
        super().__init__()
        self.task = task
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.validation_split = validation_split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.input_size = None
        self.output_size = None

    def _build_network(self):
        """Build and return the nn.Module for this model. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement _build_network()")

    def _get_input_size(self, X_np):
        """Return the feature dimension used to build the network."""
        return X_np.shape[1]

    def _get_output_size(self, y_np):
        """Return the output dimension used to build the network."""
        if self.task == 'classification':
            return int(len(np.unique(y_np)))
        if self.task == 'regression':
            return 1
        return self.input_size  # reconstruction

    def _get_loss_fn(self):
        if self.task == 'classification':
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _get_optimizer_obj(self):
        params = self.network.parameters()
        if self.optimizer == 'adam':
            return torch.optim.Adam(params, lr=self.learning_rate)
        if self.optimizer == 'sgd':
            return torch.optim.SGD(params, lr=self.learning_rate)
        if self.optimizer == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.learning_rate)
        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _get_scheduler_obj(self, opt):
        if self.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        if self.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        if self.scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        return None

    def _to_tensors(self, X_np, y_np):
        X_t = torch.tensor(X_np, dtype=torch.float32)
        if self.task == 'classification':
            y_t = torch.tensor(y_np, dtype=torch.long)
        elif self.task == 'regression':
            y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)
        else:  # reconstruction
            y_t = X_t.clone()
        return X_t, y_t

    def fit(self, X, y=None):
        """Train the model on the given data.

        Args:
            X: Training features, shape (n_samples, ...).
            y: Target labels or values. Ignored for reconstruction tasks.

        Returns:
            self: The fitted model.
        """
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y) if y is not None else X_np

        self.input_size = self._get_input_size(X_np)
        self.output_size = self._get_output_size(y_np)

        self.network = self._build_network().to(self.device)

        X_t, y_t = self._to_tensors(X_np, y_np)

        n = len(X_t)
        use_val = self.validation_split > 0 and self.patience is not None
        n_val = int(n * self.validation_split) if use_val else 0

        if n_val > 0:
            X_train, X_val = X_t[: n - n_val], X_t[n - n_val :]
            y_train, y_val = y_t[: n - n_val], y_t[n - n_val :]
        else:
            X_train, y_train = X_t, y_t
            X_val = y_val = None

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        loss_fn = self._get_loss_fn()
        opt = self._get_optimizer_obj()
        sched = self._get_scheduler_obj(opt)

        best_val_loss = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            self.network.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                loss_fn(self.network(X_b), y_b).backward()
                opt.step()

            if X_val is not None:
                self.network.eval()
                with torch.no_grad():
                    val_out = self.network(X_val.to(self.device))
                    val_loss = loss_fn(val_out, y_val.to(self.device)).item()

                if sched is not None:
                    sched.step(val_loss) if self.scheduler == 'plateau' else sched.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            elif sched is not None and self.scheduler != 'plateau':
                sched.step()

        self.is_fitted = True
        return self

    def predict(self, X):
        """Run inference and return predictions as a numpy array.

        Args:
            X: Input features, shape (n_samples, ...).

        Returns:
            predictions: Predicted values, shape (n_samples,) or (n_samples, output_size).
        """
        self._check_is_fitted()
        X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32).to(self.device)
        self.network.eval()
        with torch.no_grad():
            out = self.network(X_t)
        if self.task == 'classification':
            return out.argmax(dim=1).cpu().numpy()
        if self.task == 'regression':
            return out.squeeze(-1).cpu().numpy()
        return out.cpu().numpy()  # reconstruction

    def score(self, X, y=None):
        """Compute a performance metric on the given data.

        Returns accuracy for classification, R-squared for regression,
        and mean reconstruction MSE for reconstruction tasks.

        Args:
            X: Input features, shape (n_samples, ...).
            y: True labels or values. Ignored for reconstruction tasks.

        Returns:
            score: Performance metric as a Python float.
        """
        self._check_is_fitted()
        pred = self.predict(X)
        if self.task == 'classification':
            return float(np.mean(pred == np.array(y)))
        if self.task == 'regression':
            y_np = np.array(y, dtype=np.float32)
            ss_res = np.sum((y_np - pred) ** 2)
            ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        X_np = np.array(X, dtype=np.float32)
        return float(np.mean((X_np - pred) ** 2))
