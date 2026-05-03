from pydantic import BaseModel


class ColumnMeta(BaseModel):
    name: str
    dtype: str
    null_count: int


class DatasetResponse(BaseModel):
    id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnMeta]
    preview: list[dict]


class DatasetListItem(BaseModel):
    id: str
    filename: str
    n_rows: int
    n_cols: int
    created_at: str


class DatasetDetail(BaseModel):
    id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnMeta]
    file_path: str
    created_at: str
