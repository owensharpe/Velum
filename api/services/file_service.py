"""
Title: file_service.py
Author: Owen Sharpe
Description: On-disk storage helpers for uploaded CSV datasets and pickled
trained models, plus DataFrame column-metadata extraction.
"""

import json
import os
import uuid

import pandas as pd

DATASETS_DIR = "api/storage/datasets"
MODELS_DIR = "api/storage/models"


def save_uploaded_csv(file_content: bytes, filename: str) -> tuple[str, str, pd.DataFrame]:
    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(DATASETS_DIR, f"{dataset_id}.csv")
    with open(file_path, "wb") as f:
        f.write(file_content)
    df = pd.read_csv(file_path)
    return dataset_id, file_path, df


def extract_column_metadata(df: pd.DataFrame) -> str:
    meta = []
    for col in df.columns:
        meta.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
        })
    return json.dumps(meta)


def delete_file(path: str):
    if os.path.exists(path):
        os.remove(path)
