"""AI chat REST endpoints for LUCID Central Command.

Endpoints:
    POST /api/ai/chat      Send a message to the AI workflow agent; returns the
                           assistant's response and any tool calls made.
    GET  /api/ai/history   Return the conversation history for a given session_id.

The AI agent (``AIWorkflowAgent``) is instantiated once at startup and attached
to ``app.state.ai_agent``.  It connects to Ollama via LangChain and uses
LangGraph's ReAct loop to dispatch tasks to online specialist components via
MQTT ``cmd/task`` commands.

Timeout behaviour
-----------------
If Ollama does not respond within ``AI_CHAT_TIMEOUT`` seconds (default 60), the
endpoint returns HTTP 503.  The caller should display a user-friendly message
and suggest checking that the Ollama service is running.
"""
import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import db as DB

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for the AI chat endpoint."""
    message: str
    session_id: str = "default"


@router.post("/api/ai/chat")
async def ai_chat(body: ChatRequest, request: Request):
    """Send a message to the AI workflow agent and return its response.

    The agent persists the conversation turn to Postgres via
    ``DB.save_conversation_turns`` so history survives restarts.

    Args:
        body:    ChatRequest with ``message`` and ``session_id``.
        request: FastAPI Request; used to access ``app.state.ai_agent``.

    Returns:
        Dict with keys ``response`` (str) and ``tool_calls`` (list).

    Raises:
        HTTPException(503): If the agent times out or Ollama is unreachable.
    """
    agent = request.app.state.ai_agent
    try:
        result = await agent.chat(body.message, body.session_id)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI agent timed out — is Ollama running?")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI agent error: {e}")


@router.get("/api/ai/sessions")
async def ai_sessions():
    """Return all AI chat sessions with preview of first message."""
    return DB.list_conversations()


@router.get("/api/ai/history")
async def ai_history(session_id: str, request: Request):
    """Return the conversation history for ``session_id``.

    Args:
        session_id: Conversation identifier (matches the one used in ``/chat``).
        request:    FastAPI Request (unused, kept for consistency).

    Returns:
        Dict with key ``turns``: list of ``{"role": str, "content": str}``.
    """
    turns = DB.get_conversation_turns(session_id)
    return {"turns": [{"role": role, "content": content} for role, content in turns]}
