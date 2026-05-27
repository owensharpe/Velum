"""
Title: datasets.py
Author: Owen Sharpe
Description: FastAPI router for user-uploaded datasets under /api/v1/datasets,
exposing CSV upload, list, detail, and delete endpoints.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session, select

from api.database import Dataset, get_session
from api.schemas.dataset import DatasetDetail, DatasetListItem, DatasetResponse
from api.services.file_service import (
    delete_file,
    extract_column_metadata,
    save_uploaded_csv,
)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile, session: Session = Depends(get_session)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    dataset_id, file_path, df = save_uploaded_csv(content, file.filename)
    column_metadata = extract_column_metadata(df)

    dataset = Dataset(
        id=dataset_id,
        filename=file.filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        column_metadata=column_metadata,
        file_path=file_path,
    )
    session.add(dataset)
    session.commit()

    columns = json.loads(column_metadata)
    preview = df.head(5).to_dict(orient="records")

    return DatasetResponse(
        id=dataset_id,
        filename=file.filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        preview=preview,
    )


@router.get("", response_model=list[DatasetListItem])
def list_datasets(session: Session = Depends(get_session)):
    datasets = session.exec(select(Dataset)).all()
    return [
        DatasetListItem(
            id=d.id,
            filename=d.filename,
            n_rows=d.n_rows,
            n_cols=d.n_cols,
            created_at=d.created_at.isoformat(),
        )
        for d in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str, session: Session = Depends(get_session)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetDetail(
        id=dataset.id,
        filename=dataset.filename,
        n_rows=dataset.n_rows,
        n_cols=dataset.n_cols,
        columns=json.loads(dataset.column_metadata),
        file_path=dataset.file_path,
        created_at=dataset.created_at.isoformat(),
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, session: Session = Depends(get_session)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    delete_file(dataset.file_path)
    session.delete(dataset)
    session.commit()
    return {"detail": "Dataset deleted"}
