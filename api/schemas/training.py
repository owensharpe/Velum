from pydantic import BaseModel


class TrainRequest(BaseModel):
    dataset_id: str
    model_type: str
    target_column: str
    feature_columns: list[str]
    hyperparameters: dict = {}
    task: str  # classification | regression


class TrainResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    status: str
    hyperparameters: dict
    target_column: str
    feature_columns: list[str]
    model_id: str | None
    error_message: str | None
    created_at: str
