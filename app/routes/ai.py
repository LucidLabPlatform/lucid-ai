"""AI chat REST endpoints for LUCID Central Command.

Endpoints:
    POST /api/ai/chat          Send a message (returns full response).
    POST /api/ai/chat/stream   Send a message (returns SSE event stream).
    GET  /api/ai/sessions      List all conversation sessions.
    DELETE /api/ai/sessions/X  Delete a conversation session.
    GET  /api/ai/history       Return conversation history for a session.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import db as DB

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for the AI chat endpoint."""
    message: str
    session_id: str = "default"


@router.post("/api/ai/chat")
async def ai_chat(body: ChatRequest, request: Request):
    """Send a message to the AI agent and return its response."""
    agent = request.app.state.ai_agent
    try:
        result = await agent.chat(body.message, body.session_id)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="AI agent timed out — is Ollama running?")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI agent error: {e}")


@router.post("/api/ai/chat/stream")
async def ai_chat_stream(body: ChatRequest, request: Request):
    """Send a message and stream the response via Server-Sent Events."""
    agent = request.app.state.ai_agent

    async def event_generator():
        try:
            async for event in agent.chat_stream(body.message, body.session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'AI agent timed out'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/ai/sessions")
async def ai_sessions():
    """Return all AI chat sessions with preview of first message."""
    return DB.list_conversations()


@router.delete("/api/ai/sessions/{session_id}")
async def ai_delete_session(session_id: str):
    """Delete a conversation session and all its turns."""
    DB.delete_conversation(session_id)
    return {"deleted": session_id}


@router.get("/api/ai/history")
async def ai_history(session_id: str, request: Request):
    """Return the conversation history for a session."""
    turns = DB.get_conversation_turns(session_id)
    return {"turns": [{"role": role, "content": content} for role, content in turns]}
