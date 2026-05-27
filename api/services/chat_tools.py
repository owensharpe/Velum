"""
Title: chat_tools.py
Author: Owen Sharpe
Description: Read-only tools exposed to the Gemini chat LLM, comprising
FunctionDeclaration schemas advertised to the model and Python
implementations dispatched by execute_tool() whose outputs are trimmed
for LLM consumption.
"""

import json
from typing import Any

from sqlmodel import Session, select

from api.database import Dataset
from api.routers.models import AVAILABLE_MODELS

# tool schemas (sent to Gemini)
TOOL_DECLARATIONS = [
    {
        "name": "list_datasets",
        "description": (
            "List all datasets the user has uploaded. Returns id, filename, "
            "row count, and column count for each. Use this when the user asks "
            "what data is available or refers to a dataset without specifying which one."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_dataset_info",
        "description": (
            "Get column-level details for a specific dataset: column names, dtypes, "
            "and null counts. Use this BEFORE recommending a model or training, "
            "so you can understand what's in the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID, obtained from list_datasets.",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "list_available_models",
        "description": (
            "List all model types the user can train, with their category "
            "(classical or deep) and supported tasks (classification, regression, "
            "clustering, etc.). Use this when the user asks what models are "
            "available, or when you need to pick an appropriate model for a task."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


# tool implementations
def _list_datasets(db: Session) -> dict[str, Any]:
    datasets = db.exec(select(Dataset)).all()
    return {
        "datasets": [
            {
                "id": d.id,
                "filename": d.filename,
                "n_rows": d.n_rows,
                "n_cols": d.n_cols,
            }
            for d in datasets
        ],
        "count": len(datasets),
    }


def _get_dataset_info(db: Session, dataset_id: str) -> dict[str, Any]:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        return {"error": f"No dataset found with id '{dataset_id}'."}

    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "n_rows": dataset.n_rows,
        "n_cols": dataset.n_cols,
        "columns": json.loads(dataset.column_metadata),
    }


def _list_available_models() -> dict[str, Any]:
    return {
        "models": [
            {
                "name": m.name,
                "class_name": m.class_name,
                "category": m.category,
                "tasks": m.tasks,
            }
            for m in AVAILABLE_MODELS
        ],
        "count": len(AVAILABLE_MODELS),
    }


# dispatcher
def execute_tool(name: str, args: dict[str, Any], db: Session) -> dict[str, Any]:
    """
    Run a tool by name. Returns a JSON-serializable dictionary that will be sent
    back to Gemini as a function_response.

    Tool errors are returned as {"error": ...} rather than raised, so the
    model can reason about them and recover (e.g. ask the user to clarify).
    """
    if name == "list_datasets":
        return _list_datasets(db)

    if name == "get_dataset_info":
        dataset_id = args.get("dataset_id")
        if not dataset_id:
            return {"error": "Missing required argument: dataset_id"}
        return _get_dataset_info(db, dataset_id)

    if name == "list_available_models":
        return _list_available_models()

    return {"error": f"Unknown tool: {name}"}
