import json
import os
import uuid

import joblib
import numpy as np
import pandas as pd
from sqlmodel import Session

from api.database import TrainedModel, TrainingJob, engine
from api.services.file_service import MODELS_DIR

MODEL_REGISTRY = {
    "LinearRegression": ("velum", "LinearRegression"),
    "LogisticRegression": ("velum", "LogisticRegression"),
    "DecisionTree": ("velum", "DecisionTree"),
    "RandomForest": ("velum", "RandomForest"),
    "KNN": ("velum", "KNN"),
    "SVM": ("velum", "SVM"),
    "NaiveBayes": ("velum", "NaiveBayes"),
    "GradientBoosting": ("velum", "GradientBoosting"),
    "RidgeRegression": ("velum", "RidgeRegression"),
    "LassoRegression": ("velum", "LassoRegression"),
    "ElasticNetRegression": ("velum", "ElasticNetRegression"),
    "KMeans": ("velum", "KMeans"),
    "PCA": ("velum", "PCA"),
    "MLP": ("velum", "MLP"),
    "AutoEncoder": ("velum", "AutoEncoder"),
    "LSTM": ("velum", "LSTM"),
    "CNN": ("velum", "CNN"),
    "Transformer": ("velum", "Transformer"),
    "TabTransformer": ("velum", "TabTransformer"),
}

DEEP_MODELS = {"MLP", "AutoEncoder", "LSTM", "CNN", "Transformer", "TabTransformer"}


def _instantiate_model(model_type: str, hyperparameters: dict):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    import velum
    cls = getattr(velum, model_type)
    return cls(**hyperparameters)


def run_training(job_id: str, dataset_path: str, model_type: str,
                 target_column: str, feature_columns: list[str],
                 hyperparameters: dict, task: str, dataset_id: str,
                 progress_store: dict):
    try:
        df = pd.read_csv(dataset_path)
        X = df[feature_columns].values.astype(np.float32)
        y = df[target_column].values

        model = _instantiate_model(model_type, hyperparameters)

        is_deep = model_type in DEEP_MODELS

        if is_deep and hasattr(model, 'epochs'):
            _run_deep_training_with_progress(
                model, X, y, job_id, progress_store
            )
        else:
            model.fit(X, y)
            progress_store[job_id] = [{"event": "progress", "data": {"progress": 100}}]

        if task == "classification":
            preds = model.predict(X)
            accuracy = float(np.mean(preds == y))
            metrics = {"accuracy": accuracy}
        else:
            preds = model.predict(X)
            ss_res = np.sum((y.astype(np.float32) - preds) ** 2)
            ss_tot = np.sum((y.astype(np.float32) - np.mean(y.astype(np.float32))) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            metrics = {"r2": r2}

        model_id = str(uuid.uuid4())
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump(model, model_path)

        with Session(engine) as session:
            trained = TrainedModel(
                id=model_id,
                job_id=job_id,
                dataset_id=dataset_id,
                model_type=model_type,
                task=task,
                metrics=json.dumps(metrics),
                file_path=model_path,
            )
            session.add(trained)

            job = session.get(TrainingJob, job_id)
            job.status = "complete"
            job.model_id = model_id
            session.commit()

        events = progress_store.get(job_id, [])
        events.append({
            "event": "complete",
            "data": {"model_id": model_id, "metrics": metrics},
        })
        progress_store[job_id] = events

    except Exception as e:
        with Session(engine) as session:
            job = session.get(TrainingJob, job_id)
            job.status = "failed"
            job.error_message = str(e)
            session.commit()

        events = progress_store.get(job_id, [])
        events.append({"event": "error", "data": {"message": str(e)}})
        progress_store[job_id] = events


def _run_deep_training_with_progress(model, X, y, job_id, progress_store):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y)

    model.input_size = model._get_input_size(X_np)
    model.output_size = model._get_output_size(y_np)
    model.network = model._build_network().to(model.device)

    X_t, y_t = model._to_tensors(X_np, y_np)

    n = len(X_t)
    use_val = model.validation_split > 0 and model.patience is not None
    n_val = int(n * model.validation_split) if use_val else 0

    if n_val > 0:
        X_train, X_val = X_t[: n - n_val], X_t[n - n_val:]
        y_train, y_val = y_t[: n - n_val], y_t[n - n_val:]
    else:
        X_train, y_train = X_t, y_t
        X_val = y_val = None

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=model.batch_size, shuffle=True)
    loss_fn = model._get_loss_fn()
    opt = model._get_optimizer_obj()
    sched = model._get_scheduler_obj(opt)

    best_val_loss = float('inf')
    patience_counter = 0
    events = []

    for epoch in range(model.epochs):
        model.network.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(model.device), y_b.to(model.device)
            opt.zero_grad()
            loss = loss_fn(model.network(X_b), y_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        event_data = {"epoch": epoch + 1, "loss": round(avg_loss, 6)}

        val_loss = None
        if X_val is not None:
            model.network.eval()
            with torch.no_grad():
                val_out = model.network(X_val.to(model.device))
                val_loss = loss_fn(val_out, y_val.to(model.device)).item()
            event_data["val_loss"] = round(val_loss, 6)

            if sched is not None:
                sched.step(val_loss) if model.scheduler == 'plateau' else sched.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if model.patience and patience_counter >= model.patience:
                    events.append({"event": "progress", "data": event_data})
                    break
        elif sched is not None and model.scheduler != 'plateau':
            sched.step()

        events.append({"event": "progress", "data": event_data})

    progress_store[job_id] = events
    model.is_fitted = True
