"""
Title: training.py
Author: Owen Sharpe
Description: FastAPI router for kicking off model training as a background
task and observing progress via a Server-Sent Events stream or a status
poll endpoint.
"""

import asyncio
import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from api.database import Dataset, TrainingJob, get_session
from api.schemas.training import JobStatus, TrainRequest, TrainResponse
from api.services.training_service import DEEP_MODELS, run_training

router = APIRouter(prefix="/api/v1/train", tags=["training"])

progress_store: dict[str, list[dict]] = {}


@router.post("", response_model=TrainResponse)
def start_training(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    dataset = session.get(Dataset, req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    job_id = str(uuid.uuid4())
    job = TrainingJob(
        id=job_id,
        dataset_id=req.dataset_id,
        model_type=req.model_type,
        status="running",
        hyperparameters=json.dumps(req.hyperparameters),
        target_column=req.target_column,
        feature_columns=json.dumps(req.feature_columns),
    )
    session.add(job)
    session.commit()

    hp = dict(req.hyperparameters)
    if req.model_type in DEEP_MODELS and req.model_type != "AutoEncoder":
        hp.setdefault("task", req.task)

    background_tasks.add_task(
        run_training,
        job_id=job_id,
        dataset_path=dataset.file_path,
        model_type=req.model_type,
        target_column=req.target_column,
        feature_columns=req.feature_columns,
        hyperparameters=hp,
        task=req.task,
        dataset_id=req.dataset_id,
        progress_store=progress_store,
    )

    return TrainResponse(job_id=job_id, status="started")


@router.get("/{job_id}/stream")
async def stream_training(job_id: str, session: Session = Depends(get_session)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    async def event_generator():
        sent = 0
        while True:
            events = progress_store.get(job_id, [])
            while sent < len(events):
                ev = events[sent]
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
                sent += 1
                if ev["event"] in ("complete", "error"):
                    return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, session: Session = Depends(get_session)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return JobStatus(
        id=job.id,
        dataset_id=job.dataset_id,
        model_type=job.model_type,
        status=job.status,
        hyperparameters=json.loads(job.hyperparameters),
        target_column=job.target_column,
        feature_columns=json.loads(job.feature_columns),
        model_id=job.model_id,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
    )
