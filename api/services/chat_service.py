"""
Title: chat_service.py
Author: Owen Sharpe
Description: Chat orchestration service that runs a manual Gemini tool-call
loop over a session's ChatMessage history, dispatching function calls
through chat_tools and persisting every user, assistant, and tool turn
so sessions are fully resumable.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from google import genai
from google.genai import types
from sqlmodel import Session, select

from api.database import ChatMessage, ChatSession
from api.services.chat_tools import TOOL_DECLARATIONS, execute_tool

# config
MODEL_NAME = "gemini-2.5-flash"
MAX_TOOL_CALLS_PER_TURN = 8

SYSTEM_PROMPT = """You are Velum's ML assistant. You help users explore their \
uploaded datasets and train machine learning models.

Hard rules — these are not suggestions:
- You can ONLY take actions by calling the provided tools. You have no other \
capabilities.
- If a user asks for something your tools don't support, say so plainly. Do \
not invent capabilities or pretend to do things you cannot.
- Tool names are internal. Never mention tool names to the user or instruct \
them to "call" anything. The user cannot call tools — only you can.
- Gather context before asking the user. If you need information that a tool \
can give you, CALL THE TOOL. Do not offer to do it, do not ask permission — \
just call it. The user expects you to investigate before asking them \
questions.
- Before recommending a model or training, call get_dataset_info on the \
relevant dataset. Do not guess at column names.
- Only ask the user a clarifying question AFTER you've used your tools to \
gather all the context you can. The question should be about something the \
tools cannot answer (like the user's intent or goal).
- Be concise. Users are technical; skip filler. Also, be very amiable!

Example of correct behavior:
  User: "Train a model for me"
  You: [call list_datasets, see there's one dataset called sales.csv]
       [call get_dataset_info on it, see the columns]
       "I see your dataset has columns X, Y, Z. Which would you like to \
predict, and is this classification or regression?"

NOT:
  User: "Train a model for me"
  You: "Which dataset would you like to use?" ← wrong, you should have looked first."""

# client (lazy-initialized so import doesn't require the env var to be set)
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# tool config built from chat_tools.TOOL_DECLARATIONS
def _build_tool_config() -> types.Tool:
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters_json_schema=t["parameters"],
        )
        for t in TOOL_DECLARATIONS
    ]
    return types.Tool(function_declarations=declarations)


"""
DB <-> Gemini message conversion

We store ChatMessage.content as a JSON string. The structure we store
mirrors Gemini's Part structure so reconstruction is trivial:

    user message:   [{"type": "text", "text": "..."}]
    assistant text: [{"type": "text", "text": "..."}]
    assistant call: [{"type": "function_call", "name": "...", "args": {...}}]
    tool result:    [{"type": "function_response", "name": "...",
                      "response": {...}}]

Role maps:  DB user -> Gemini "user",  DB assistant -> Gemini "model",
            DB tool -> Gemini "user"  (Gemini expects function responses
                                       under the "user" role).
"""


def _db_to_gemini_content(msg: ChatMessage) -> types.Content:
    parts_data = json.loads(msg.content)
    parts: list[types.Part] = []

    for p in parts_data:
        if p["type"] == "text":
            parts.append(types.Part(text=p["text"]))
        elif p["type"] == "function_call":
            parts.append(types.Part(
                function_call=types.FunctionCall(name=p["name"], args=p["args"])
            ))
        elif p["type"] == "function_response":
            parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=p["name"], response=p["response"]
                )
            ))

    role = "model" if msg.role == "assistant" else "user"
    return types.Content(role=role, parts=parts)


def _persist_message(
    db: Session, session_id: str, role: str, parts: list[dict[str, Any]]
) -> None:
    db.add(ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=json.dumps(parts),
    ))
    db.commit()


# public API
def get_or_create_session(db: Session, session_id: Optional[str]) -> ChatSession:
    if session_id:
        existing = db.get(ChatSession, session_id)
        if existing:
            return existing

    new_session = ChatSession(id=str(uuid.uuid4()))
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def send_message(
    db: Session, session_id: str, user_text: str
) -> dict[str, Any]:

    # load history (user/assistant/tool messages, in order)
    history_msgs = db.exec(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    ).all()
    history = [_db_to_gemini_content(m) for m in history_msgs]

    # persist and add the new user message
    user_parts = [{"type": "text", "text": user_text}]
    _persist_message(db, session_id, "user", user_parts)
    history.append(types.Content(
        role="user", parts=[types.Part(text=user_text)]
    ))

    client = _get_client()
    tool_config = _build_tool_config()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool_config],
    )

    tool_calls_made = 0
    final_text: Optional[str] = None

    # Gemini may emit function_calls in sequence; we execute each
    # and feed results back until it produces text (or we hit the cap)
    while tool_calls_made < MAX_TOOL_CALLS_PER_TURN:
        response = client.models.generate_content(
            model=MODEL_NAME, contents=history, config=config
        )

        candidate_parts = response.candidates[0].content.parts
        text_chunks: list[str] = []
        function_calls: list[types.FunctionCall] = []

        for part in candidate_parts:
            if part.text:
                text_chunks.append(part.text)
            if part.function_call:
                function_calls.append(part.function_call)

        # persist the assistant turn (may contain text, calls, or both)
        assistant_parts: list[dict[str, Any]] = []
        for part in candidate_parts:
            if part.text:
                assistant_parts.append({"type": "text", "text": part.text})
            if part.function_call:
                assistant_parts.append({
                    "type": "function_call",
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args or {}),
                })

        if assistant_parts:
            _persist_message(db, session_id, "assistant", assistant_parts)
            history.append(response.candidates[0].content)

        # no tool calls means the model is done; capture text and exit
        if not function_calls:
            final_text = "".join(text_chunks).strip() or "(no response)"
            break

        # execute each tool call, persist results, and feed back to Gemini
        tool_response_parts: list[types.Part] = []
        tool_response_db_parts: list[dict[str, Any]] = []

        for call in function_calls:
            tool_calls_made += 1
            args = dict(call.args or {})
            result = execute_tool(call.name, args, db)

            tool_response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=call.name, response=result
                )
            ))
            tool_response_db_parts.append({
                "type": "function_response",
                "name": call.name,
                "response": result,
            })

        _persist_message(db, session_id, "tool", tool_response_db_parts)
        history.append(types.Content(role="user", parts=tool_response_parts))

    if final_text is None:
        # hit the cap without resolution
        final_text = (
            "I hit the maximum number of tool calls for this turn without "
            "reaching an answer. Could you simplify the request or ask again?"
        )
        _persist_message(
            db, session_id, "assistant",
            [{"type": "text", "text": final_text}],
        )

    # bump session.updated_at so most-recent-first sorting works in the UI
    chat_session = db.get(ChatSession, session_id)
    if chat_session:
        chat_session.updated_at = datetime.utcnow()
        db.add(chat_session)
        db.commit()

    return {"session_id": session_id, "reply": final_text}
