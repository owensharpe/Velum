"""
Title: chat_tools.py
Author: Owen Sharpe
Description: Tools exposed to the Gemini chat LLM, comprising
FunctionDeclaration schemas advertised to the model and Python
implementations dispatched by execute_tool() whose outputs are trimmed
for LLM consumption.
"""

import json
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from api.database import Dataset
from api.routers.models import AVAILABLE_MODELS
from api.services.training_advisor import (
    recommend_model,
    default_hyperparameters,
)

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
    {
        "name": "propose_training",
        "description": (
            "Generate a training plan for a model on a given dataset. This is "
            "a PROPOSAL only — it does NOT start training. Use this whenever "
            "the user wants to train a model. After calling, present the plan "
            "to the user in plain prose and WAIT for explicit approval before "
            "calling start_training. If the user modifies any part of the plan, "
            "call propose_training again with the changes — do not pass altered "
            "arguments to start_training."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID, from list_datasets.",
                },
                "target_column": {
                    "type": "string",
                    "description": "Name of the column to predict.",
                },
                "user_hint": {
                    "type": "string",
                    "description": (
                        "Optional free-text hint about what the user wants "
                        "(e.g. 'fast', 'most accurate', 'interpretable'). "
                        "Empty string if none."
                    ),
                },
                "model_override": {
                    "type": "string",
                    "description": (
                        "Optional: if the user explicitly named a model type "
                        "(from list_available_models), pass its class_name "
                        "here. Empty string to let the system choose."
                    ),
                },
                "hyperparameter_overrides": {
                    "type": "object",
                    "description": (
                        "Optional: any hyperparameters the user specified "
                        "explicitly. Empty object {} to use defaults."
                    ),
                },
            },
            "required": ["dataset_id", "target_column"],
        },
    },
    {
        "name": "start_training",
        "description": (
            "Start a training job. ONLY call this AFTER the user has explicitly "
            "approved a plan returned by propose_training. Arguments must match "
            "the approved plan exactly. If the user wants any change, call "
            "propose_training again instead of altering arguments here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "model_class_name": {
                    "type": "string",
                    "description": "The class_name from list_available_models.",
                },
                "target_column": {"type": "string"},
                "hyperparameters": {"type": "object"},
                "train_test_split": {
                    "type": "number",
                    "description": "Fraction of data used for training, e.g. 0.8.",
                },
            },
            "required": [
                "dataset_id",
                "model_class_name",
                "target_column",
                "hyperparameters",
                "train_test_split",
            ],
        },
    },
    {
        "name": "get_training_status",
        "description": (
            "Check the status of a training job. Returns status "
            "(queued/running/done/failed), progress info if running, "
            "and final metrics if complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "training_job_id": {"type": "string"},
            },
            "required": ["training_job_id"],
        },
    },
    {
        "name": "list_training_jobs",
        "description": (
            "List recent training jobs, most recent first. Useful for "
            "answering questions about previously trained models."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of jobs to return. Default 10.",
                },
            },
        },
    },
]


# tool implementations — read-only
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


# tool implementations — training
def _detect_task_type(
    target_series: pd.Series, target_column: str
) -> tuple[str | None, str | None]:
    """
    Returns (task_type, error). Exactly one is non-None.
    task_type is "classification" or "regression".
    """
    non_null = target_series.dropna()
    n_unique = non_null.nunique()

    if n_unique == 0:
        return None, f"Target column '{target_column}' has no non-null values."
    if n_unique == 1:
        return None, (
            f"Target column '{target_column}' has only one unique value — "
            "there is nothing to predict."
        )

    if pd.api.types.is_numeric_dtype(non_null):
        # numeric with very few unique values is probably categorical
        # (e.g. 0/1 labels stored as int).
        if n_unique <= 10:
            return "classification", None
        return "regression", None

    return "classification", None


def _build_warnings(
    df: pd.DataFrame, target_column: str, task_type: str
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    target = df[target_column]

    # dataset-level checks
    n_rows = len(df)
    if n_rows < 100:
        warnings.append({
            "severity": "warning",
            "message": (
                f"This dataset is very small ({n_rows} rows). "
                "Results may not generalize well."
            ),
        })

    # target-column checks
    n_null = int(target.isnull().sum())
    if n_null > 0:
        warnings.append({
            "severity": "info",
            "message": (
                f"Target column has {n_null} null values; those rows will "
                "be dropped during training."
            ),
        })

    if task_type == "classification":
        counts = target.dropna().value_counts(normalize=True)
        if len(counts) > 0:
            minority_pct = counts.min() * 100
            if minority_pct < 10:
                majority_pct = counts.max() * 100
                warnings.append({
                    "severity": "warning",
                    "message": (
                        f"Class imbalance: majority class is "
                        f"{majority_pct:.0f}% of data, minority is "
                        f"{minority_pct:.0f}%. Accuracy alone may be misleading."
                    ),
                })

        n_unique = target.dropna().nunique()
        if n_unique > 50:
            warnings.append({
                "severity": "warning",
                "message": (
                    f"Target has {n_unique} unique values. This is unusually "
                    "high for classification — is this actually a regression "
                    "problem?"
                ),
            })

    # feature-column checks (apply to both classification and regression)
    feature_cols = [c for c in df.columns if c != target_column]

    cols_with_nulls = {
        col: int(df[col].isnull().sum())
        for col in feature_cols
        if df[col].isnull().any()
    }
    if cols_with_nulls:
        formatted = ", ".join(
            f"'{col}' ({n})" for col, n in cols_with_nulls.items()
        )
        warnings.append({
            "severity": "warning",
            "message": (
                f"Missing values in feature columns: {formatted}. "
                "Training will fail unless these are handled — you may need "
                "to drop or fill them before approving this plan."
            ),
        })

    non_numeric_cols = [
        col for col in feature_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        formatted = ", ".join(f"'{c}'" for c in non_numeric_cols)
        warnings.append({
            "severity": "warning",
            "message": (
                f"Non-numeric feature columns detected: {formatted}. "
                "These need to be encoded numerically (e.g. one-hot) before "
                "training, or training will fail."
            ),
        })

    return warnings


def _estimate_training_time(class_name: str, n_rows: int) -> int:
    """Very rough estimate. Refine once we have real measurements."""
    if "RandomForest" in class_name:
        return max(5, n_rows // 1000)
    return max(2, n_rows // 5000)


def _propose_training(
    db: Session,
    dataset_id: str,
    target_column: str,
    user_hint: str = "",
    model_override: str = "",
    hyperparameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        return {"error": f"No dataset found with id '{dataset_id}'."}

    # load the actual data. adjust dataset.file_path to whatever your
    # dataset model exposes for the on-disk path.
    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as e:
        return {"error": f"Could not read dataset file: {e}"}

    if target_column not in df.columns:
        return {
            "error": (
                f"Column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        }

    # detect task type
    task_type, err = _detect_task_type(df[target_column], target_column)
    if err:
        return {"error": err}

    # pick the model; override if provided, otherwise heuristic
    if model_override:
        available = {m.class_name for m in AVAILABLE_MODELS}
        if model_override not in available:
            return {
                "error": (
                    f"Unknown model '{model_override}'. "
                    f"Available: {sorted(available)}"
                )
            }
        model_class_name = model_override
        rationale = f"Using {model_class_name} as requested."
    else:
        model_class_name, rationale = recommend_model(task_type, len(df))

    # assemble hyperparameters: defaults + overrides
    hyperparameters = default_hyperparameters(model_class_name)
    if hyperparameter_overrides:
        hyperparameters.update(hyperparameter_overrides)

    # warnings
    warnings = _build_warnings(df, target_column, task_type)

    return {
        "plan_summary": (
            f"Train a {model_class_name} on '{dataset.filename}' to predict "
            f"'{target_column}' ({task_type})."
        ),
        "model": {
            "class_name": model_class_name,
            "rationale": rationale,
        },
        "hyperparameters": hyperparameters,
        "hyperparameter_rationale": (
            "Default hyperparameters for this model and dataset size."
            if not hyperparameter_overrides
            else "User-overridden hyperparameters applied to defaults."
        ),
        "data": {
            "rows": len(df),
            "features": len(df.columns) - 1,  # excluding target
            "feature_columns": [c for c in df.columns if c != target_column],
            "target_column": target_column,
            "task_type": task_type,
            "train_test_split": "80/20",
        },
        "estimated_training_seconds": _estimate_training_time(
            model_class_name, len(df)
        ),
        "warnings": warnings,
    }


# stubs, which will be replaced when the rest of the training milestone lands
def _start_training_stub(
    db: Session,
    dataset_id: str,
    model_class_name: str,
    target_column: str,
    hyperparameters: dict[str, Any],
    train_test_split: float,
) -> dict[str, Any]:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        return {"error": f"No dataset found with id '{dataset_id}'."}

    return {
        "_stub": True,
        "training_job_id": "job_stub_001",
        "status": "queued",
        "message": "[STUB] Training not actually started.",
    }


def _get_training_status_stub(db: Session, training_job_id: str) -> dict[str, Any]:
    return {
        "_stub": True,
        "training_job_id": training_job_id,
        "status": "running",
        "progress": {"elapsed_seconds": 7},
        "metrics": None,
    }


def _list_training_jobs_stub(db: Session, limit: int = 10) -> dict[str, Any]:
    return {"_stub": True, "jobs": []}


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

    if name == "propose_training":
        dataset_id = args.get("dataset_id")
        target_column = args.get("target_column")
        if not dataset_id or not target_column:
            return {"error": "Missing required arguments: dataset_id, target_column"}
        return _propose_training(
            db,
            dataset_id=dataset_id,
            target_column=target_column,
            user_hint=args.get("user_hint", ""),
            model_override=args.get("model_override", ""),
            hyperparameter_overrides=args.get("hyperparameter_overrides") or {},
        )

    if name == "start_training":
        required = [
            "dataset_id", "model_class_name", "target_column",
            "hyperparameters", "train_test_split",
        ]
        missing = [k for k in required if k not in args]
        if missing:
            return {"error": f"Missing required arguments: {', '.join(missing)}"}
        return _start_training_stub(
            db,
            dataset_id=args["dataset_id"],
            model_class_name=args["model_class_name"],
            target_column=args["target_column"],
            hyperparameters=args["hyperparameters"],
            train_test_split=args["train_test_split"],
        )

    if name == "get_training_status":
        training_job_id = args.get("training_job_id")
        if not training_job_id:
            return {"error": "Missing required argument: training_job_id"}
        return _get_training_status_stub(db, training_job_id)

    if name == "list_training_jobs":
        return _list_training_jobs_stub(db, limit=args.get("limit", 10))

    return {"error": f"Unknown tool: {name}"}
