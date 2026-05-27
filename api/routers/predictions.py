"""
Title: predictions.py
Author: Owen Sharpe
Description: FastAPI router exposing POST /api/v1/models/{model_id}/predict,
which loads the joblib-pickled estimator and runs inference over an
uploaded CSV.
"""

import io

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session

from api.database import TrainedModel, get_session
from api.schemas.model import PredictionResponse

router = APIRouter(prefix="/api/v1/models", tags=["predictions"])


@router.post("/{model_id}/predict", response_model=PredictionResponse)
async def predict(model_id: str, file: UploadFile, session: Session = Depends(get_session)):
    model_record = session.get(TrainedModel, model_id)
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        model = joblib.load(model_record.file_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to load model file")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    X = df.values.astype(np.float32)
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    predictions = preds.tolist() if hasattr(preds, "tolist") else list(preds)

    return PredictionResponse(
        predictions=predictions,
        model_id=model_id,
        model_type=model_record.model_type,
    )
