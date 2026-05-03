from pydantic import BaseModel


class ModelListItem(BaseModel):
    id: str
    model_type: str
    task: str
    dataset_id: str
    created_at: str


class ModelDetail(BaseModel):
    id: str
    job_id: str
    dataset_id: str
    model_type: str
    task: str
    metrics: dict
    file_path: str
    created_at: str


class HyperparamInfo(BaseModel):
    name: str
    type: str
    default: object


class AvailableModel(BaseModel):
    name: str
    class_name: str
    category: str  # classical | deep
    tasks: list[str]
    hyperparameters: list[HyperparamInfo]


class PredictionResponse(BaseModel):
    predictions: list
    model_id: str
    model_type: str
