import json
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

DATABASE_URL = "sqlite:///api/velum.db"
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


class Dataset(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    n_rows: int
    n_cols: int
    column_metadata: str  # JSON
    file_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingJob(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(foreign_key="dataset.id")
    model_type: str
    status: str  # running | complete | failed
    hyperparameters: str  # JSON
    target_column: str
    feature_columns: str  # JSON
    model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainedModel(SQLModel, table=True):
    id: str = Field(primary_key=True)
    job_id: str = Field(foreign_key="trainingjob.id")
    dataset_id: str = Field(foreign_key="dataset.id")
    model_type: str
    task: str
    metrics: str  # JSON
    file_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(SQLModel, table=True):
    id: str = Field(primary_key=True)
    active_dataset_id: Optional[str] = Field(default=None, foreign_key="dataset.id")
    active_job_id: Optional[str] = Field(default=None, foreign_key="trainingjob.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(SQLModel, table=True):
    id: str = Field(primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id")
    role: str  # user | assistant | tool
    content: str  # JSON
    created_at: datetime = Field(default_factory=datetime.utcnow)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session