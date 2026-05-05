"""AI chat REST endpoints for LUCID Central Command.

Endpoints:
    POST /api/ai/chat            Send a message (returns full response).
    POST /api/ai/chat/stream     Send a message (returns SSE event stream).
    GET  /api/ai/sessions        List all conversation sessions.
    DELETE /api/ai/sessions      Delete every conversation session.
    DELETE /api/ai/sessions/X    Delete a conversation session.
    GET  /api/ai/history         Return conversation history for a session.
    GET  /api/ai/prompts         Return all prompt defaults + overrides.
    PUT  /api/ai/prompts/{name}  Set or clear a prompt override.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import db as DB
from app.ai import prompts as P
from app.ai.prompts_store import load_overrides, save_overrides

router = APIRouter()


# Keys exposed via the prompts API. The first six are template strings used by
# specialists; CLASSIFY_PROMPT and VOICE_SUMMARY_PROMPT are utility prompts.
# LAB_CONTEXT is plain text injected via {lab_context} into every specialist
# prompt — researchers use it to give project-specific context to the agent.
_PROMPT_KEYS = (
    "FLEET_SYSTEM_PROMPT",
    "COMMAND_SYSTEM_PROMPT",
    "EXPERIMENT_SYSTEM_PROMPT",
    "TOPIC_LINK_SYSTEM_PROMPT",
    "LOGS_SYSTEM_PROMPT",
    "CONVERSATION_SYSTEM_PROMPT",
    "CLASSIFY_PROMPT",
    "VOICE_SUMMARY_PROMPT",
    "LAB_CONTEXT",  # not a prompt, but lives in the same store
)


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


@router.delete("/api/ai/sessions")
async def ai_delete_all_sessions():
    """Delete every AI chat session and all turns."""
    count = DB.delete_all_conversations()
    return {"deleted_count": count}


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


# ── Prompts API ────────────────────────────────────────────────────────────

@router.get("/api/ai/prompts")
async def get_prompts():
    """Return defaults from prompts.py + current overrides from disk."""
    overrides = load_overrides()
    out = {}
    for key in _PROMPT_KEYS:
        default = getattr(P, key, "") if key != "LAB_CONTEXT" else ""
        out[key] = {
            "default": default,
            "override": overrides.get(key),
        }
    return out


class PromptUpdate(BaseModel):
    override: str | None = None


@router.put("/api/ai/prompts/{name}")
async def set_prompt(name: str, body: PromptUpdate):
    """Set or clear an override for a single prompt key.

    Set ``override`` to a non-empty string to override the default.
    Set ``override`` to ``null`` or an empty string to clear and use the default.
    """
    if name not in _PROMPT_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown prompt key: {name}")

    overrides = load_overrides()
    new_text = (body.override or "").strip()
    if not new_text:
        overrides.pop(name, None)
        action = "reset"
    else:
        overrides[name] = body.override
        action = "set"
    save_overrides(overrides)
    return {"ok": True, "name": name, "action": action}
