"""
Title: main.py
Author: Owen Sharpe
Description: FastAPI application entrypoint that wires up CORS, registers the
chat / datasets / training / models / predictions routers, and on startup
creates the SQLite tables plus on-disk storage directories.
"""

from dotenv import load_dotenv
load_dotenv()

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import create_db_and_tables
from api.routers import chat, datasets, models, predictions, training

app = FastAPI(title="Velum ML Platform API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(predictions.router)


@app.on_event("startup")
def on_startup():
    os.makedirs("api/storage/datasets", exist_ok=True)
    os.makedirs("api/storage/models", exist_ok=True)
    create_db_and_tables()


@app.get("/")
def root():
    return {"name": "Velum ML Platform API", "version": "0.1.0"}
