"""
Title: chat.py
Author: Owen Sharpe
Description: Pydantic request/response schemas for the chat endpoint.
"""

from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
