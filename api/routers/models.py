"""
Title: models.py
Author: Owen Sharpe
Description: FastAPI router for trained-model list/detail/delete and the
static AVAILABLE_MODELS catalog that describes every supported model and
its hyperparameters to the frontend and the chat LLM.
"""

import json

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session, select

from api.database import TrainedModel, get_session
from api.schemas.model import (
    AvailableModel,
    HyperparamInfo,
    ModelDetail,
    ModelListItem,
    PredictionResponse,
)
from api.services.file_service import delete_file

router = APIRouter(prefix="/api/v1/models", tags=["models"])

AVAILABLE_MODELS = [
    AvailableModel(
        name="Linear Regression",
        class_name="LinearRegression",
        category="classical",
        tasks=["regression"],
        hyperparameters=[],
    ),
    AvailableModel(
        name="Logistic Regression",
        class_name="LogisticRegression",
        category="classical",
        tasks=["classification"],
        hyperparameters=[
            HyperparamInfo(name="max_iter", type="int", default=1000),
        ],
    ),
    AvailableModel(
        name="Decision Tree",
        class_name="DecisionTree",
        category="classical",
        tasks=["classification"],
        hyperparameters=[
            HyperparamInfo(name="max_depth", type="int", default=None),
            HyperparamInfo(name="min_samples_split", type="int", default=2),
            HyperparamInfo(name="min_samples_leaf", type="int", default=1),
            HyperparamInfo(name="criterion", type="str", default="gini"),
        ],
    ),
    AvailableModel(
        name="Random Forest",
        class_name="RandomForest",
        category="classical",
        tasks=["classification"],
        hyperparameters=[
            HyperparamInfo(name="n_estimators", type="int", default=100),
            HyperparamInfo(name="max_depth", type="int", default=None),
            HyperparamInfo(name="random_state", type="int", default=None),
        ],
    ),
    AvailableModel(
        name="K-Nearest Neighbors",
        class_name="KNN",
        category="classical",
        tasks=["classification"],
        hyperparameters=[
            HyperparamInfo(name="n_neighbors", type="int", default=5),
        ],
    ),
    AvailableModel(
        name="Support Vector Machine",
        class_name="SVM",
        category="classical",
        tasks=["classification"],
        hyperparameters=[
            HyperparamInfo(name="kernel", type="str", default="rbf"),
            HyperparamInfo(name="C", type="float", default=1.0),
        ],
    ),
    AvailableModel(
        name="Naive Bayes",
        class_name="NaiveBayes",
        category="classical",
        tasks=["classification"],
        hyperparameters=[],
    ),
    AvailableModel(
        name="Gradient Boosting",
        class_name="GradientBoosting",
        category="classical",
        tasks=["classification"],
        hyperparameters=[
            HyperparamInfo(name="n_estimators", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.1),
            HyperparamInfo(name="max_depth", type="int", default=3),
            HyperparamInfo(name="random_state", type="int", default=None),
        ],
    ),
    AvailableModel(
        name="Ridge Regression",
        class_name="RidgeRegression",
        category="classical",
        tasks=["regression"],
        hyperparameters=[
            HyperparamInfo(name="alpha", type="float", default=1.0),
        ],
    ),
    AvailableModel(
        name="Lasso Regression",
        class_name="LassoRegression",
        category="classical",
        tasks=["regression"],
        hyperparameters=[
            HyperparamInfo(name="alpha", type="float", default=1.0),
            HyperparamInfo(name="max_iter", type="int", default=1000),
        ],
    ),
    AvailableModel(
        name="Elastic Net Regression",
        class_name="ElasticNetRegression",
        category="classical",
        tasks=["regression"],
        hyperparameters=[
            HyperparamInfo(name="alpha", type="float", default=1.0),
            HyperparamInfo(name="l1_ratio", type="float", default=0.5),
            HyperparamInfo(name="max_iter", type="int", default=1000),
        ],
    ),
    AvailableModel(
        name="K-Means Clustering",
        class_name="KMeans",
        category="classical",
        tasks=["clustering"],
        hyperparameters=[
            HyperparamInfo(name="n_clusters", type="int", default=8),
            HyperparamInfo(name="max_iter", type="int", default=300),
            HyperparamInfo(name="random_state", type="int", default=None),
        ],
    ),
    AvailableModel(
        name="PCA",
        class_name="PCA",
        category="classical",
        tasks=["dimensionality_reduction"],
        hyperparameters=[
            HyperparamInfo(name="n_components", type="int", default=None),
        ],
    ),
    AvailableModel(
        name="Multi-Layer Perceptron",
        class_name="MLP",
        category="deep",
        tasks=["classification", "regression"],
        hyperparameters=[
            HyperparamInfo(name="task", type="str", default="classification"),
            HyperparamInfo(name="hidden_layers", type="list", default=[128, 64]),
            HyperparamInfo(name="dropout", type="float", default=0.0),
            HyperparamInfo(name="activation", type="str", default="relu"),
            HyperparamInfo(name="epochs", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.001),
            HyperparamInfo(name="batch_size", type="int", default=32),
            HyperparamInfo(name="optimizer", type="str", default="adam"),
            HyperparamInfo(name="scheduler", type="str", default=None),
            HyperparamInfo(name="patience", type="int", default=None),
            HyperparamInfo(name="validation_split", type="float", default=0.1),
        ],
    ),
    AvailableModel(
        name="AutoEncoder",
        class_name="AutoEncoder",
        category="deep",
        tasks=["reconstruction"],
        hyperparameters=[
            HyperparamInfo(name="encoder_layers", type="list", default=None),
            HyperparamInfo(name="latent_dim", type="int", default=16),
            HyperparamInfo(name="epochs", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.001),
            HyperparamInfo(name="batch_size", type="int", default=32),
            HyperparamInfo(name="optimizer", type="str", default="adam"),
            HyperparamInfo(name="scheduler", type="str", default=None),
            HyperparamInfo(name="patience", type="int", default=None),
            HyperparamInfo(name="validation_split", type="float", default=0.1),
        ],
    ),
    AvailableModel(
        name="LSTM",
        class_name="LSTM",
        category="deep",
        tasks=["classification", "regression"],
        hyperparameters=[
            HyperparamInfo(name="task", type="str", default="classification"),
            HyperparamInfo(name="hidden_size", type="int", default=64),
            HyperparamInfo(name="num_layers", type="int", default=1),
            HyperparamInfo(name="dropout", type="float", default=0.0),
            HyperparamInfo(name="bidirectional", type="bool", default=False),
            HyperparamInfo(name="epochs", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.001),
            HyperparamInfo(name="batch_size", type="int", default=32),
            HyperparamInfo(name="optimizer", type="str", default="adam"),
            HyperparamInfo(name="scheduler", type="str", default=None),
            HyperparamInfo(name="patience", type="int", default=None),
            HyperparamInfo(name="validation_split", type="float", default=0.1),
        ],
    ),
    AvailableModel(
        name="CNN",
        class_name="CNN",
        category="deep",
        tasks=["classification", "regression"],
        hyperparameters=[
            HyperparamInfo(name="task", type="str", default="classification"),
            HyperparamInfo(name="channels", type="list", default=None),
            HyperparamInfo(name="kernel_size", type="int", default=3),
            HyperparamInfo(name="pooling", type="str", default="max"),
            HyperparamInfo(name="input_dims", type="int", default=1),
            HyperparamInfo(name="epochs", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.001),
            HyperparamInfo(name="batch_size", type="int", default=32),
            HyperparamInfo(name="optimizer", type="str", default="adam"),
            HyperparamInfo(name="scheduler", type="str", default=None),
            HyperparamInfo(name="patience", type="int", default=None),
            HyperparamInfo(name="validation_split", type="float", default=0.1),
        ],
    ),
    AvailableModel(
        name="Transformer",
        class_name="Transformer",
        category="deep",
        tasks=["classification", "regression"],
        hyperparameters=[
            HyperparamInfo(name="task", type="str", default="classification"),
            HyperparamInfo(name="d_model", type="int", default=64),
            HyperparamInfo(name="nhead", type="int", default=4),
            HyperparamInfo(name="num_encoder_layers", type="int", default=2),
            HyperparamInfo(name="dim_feedforward", type="int", default=128),
            HyperparamInfo(name="dropout", type="float", default=0.1),
            HyperparamInfo(name="epochs", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.001),
            HyperparamInfo(name="batch_size", type="int", default=32),
            HyperparamInfo(name="optimizer", type="str", default="adam"),
            HyperparamInfo(name="scheduler", type="str", default=None),
            HyperparamInfo(name="patience", type="int", default=None),
            HyperparamInfo(name="validation_split", type="float", default=0.1),
        ],
    ),
    AvailableModel(
        name="TabTransformer",
        class_name="TabTransformer",
        category="deep",
        tasks=["classification", "regression"],
        hyperparameters=[
            HyperparamInfo(name="task", type="str", default="classification"),
            HyperparamInfo(name="num_categories", type="list", default=None),
            HyperparamInfo(name="num_continuous", type="int", default=0),
            HyperparamInfo(name="d_model", type="int", default=32),
            HyperparamInfo(name="nhead", type="int", default=4),
            HyperparamInfo(name="num_layers", type="int", default=2),
            HyperparamInfo(name="dropout", type="float", default=0.1),
            HyperparamInfo(name="epochs", type="int", default=100),
            HyperparamInfo(name="learning_rate", type="float", default=0.001),
            HyperparamInfo(name="batch_size", type="int", default=32),
            HyperparamInfo(name="optimizer", type="str", default="adam"),
            HyperparamInfo(name="scheduler", type="str", default=None),
            HyperparamInfo(name="patience", type="int", default=None),
            HyperparamInfo(name="validation_split", type="float", default=0.1),
        ],
    ),
]


@router.get("", response_model=list[ModelListItem])
def list_models(session: Session = Depends(get_session)):
    models = session.exec(select(TrainedModel)).all()
    return [
        ModelListItem(
            id=m.id,
            model_type=m.model_type,
            task=m.task,
            dataset_id=m.dataset_id,
            created_at=m.created_at.isoformat(),
        )
        for m in models
    ]


@router.get("/available", response_model=list[AvailableModel])
def get_available_models():
    return AVAILABLE_MODELS


@router.get("/{model_id}", response_model=ModelDetail)
def get_model(model_id: str, session: Session = Depends(get_session)):
    model = session.get(TrainedModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelDetail(
        id=model.id,
        job_id=model.job_id,
        dataset_id=model.dataset_id,
        model_type=model.model_type,
        task=model.task,
        metrics=json.loads(model.metrics),
        file_path=model.file_path,
        created_at=model.created_at.isoformat(),
    )


@router.delete("/{model_id}")
def delete_model(model_id: str, session: Session = Depends(get_session)):
    model = session.get(TrainedModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    delete_file(model.file_path)
    session.delete(model)
    session.commit()
    return {"detail": "Model deleted"}
