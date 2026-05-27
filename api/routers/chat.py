"""
Title: chat.py
Author: Owen Sharpe
Description: FastAPI router exposing the POST /api/v1/chat endpoint, which
resolves (or creates) a ChatSession and delegates the Gemini tool-loop to
chat_service.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from api.database import get_session
from api.schemas.chat import ChatRequest, ChatResponse
from api.services.chat_service import get_or_create_session, send_message

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_session)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    chat_session = get_or_create_session(db, req.session_id)

    try:
        result = send_message(db, chat_session.id, req.message)
    except RuntimeError as e:
        # likely that GEMINI_API_KEY not set.
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(**result)
