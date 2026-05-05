"""AI Workflow Supervisor Agent for LUCID Central Command.

This module implements the multi-specialist agent system that powers the
``/ai`` chat interface. The agent acts as a **supervisor** that:

1. Loads the conversation history from Postgres for the given ``session_id``.
2. Classifies the user's intent via a two-tier classifier (keywords + LLM).
3. Routes to the appropriate specialist agent (fleet, command, experiment,
   topic_link, logs, or conversation).
4. Each specialist runs a focused ReAct loop with only its domain's tools.
5. Persists the user+assistant exchange to Postgres.

Environment variables:
    OLLAMA_MODEL        Ollama model tag (default: ``"qwen3:14b"``).
    OLLAMA_BASE_URL     Ollama HTTP base URL (default: ``"http://ollama:11434"``).

Key classes:
    AIWorkflowAgent — instantiated once at startup; graph compiled once.
"""

import json
import logging
import os
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from app import db as DB
from app.ai.graph import build_graph

log = logging.getLogger(__name__)


class AIWorkflowAgent:
    """Multi-specialist LangGraph agent for LUCID fleet management.

    The LangGraph StateGraph is compiled once at ``__init__`` with all
    specialist agents pre-built. Per-request, only fleet context and
    conversation history are injected.

    Attributes:
        _fleet:  HTTP client for ``lucid-orchestrator`` command dispatch.
        _llm:    Configured ``ChatOllama`` language model instance.
        _graph:  Compiled LangGraph StateGraph.
    """

    def __init__(self, fleet):
        self._fleet = fleet
        self._llm = ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "qwen3:14b"),
            temperature=0,
            num_ctx=32768,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        )
        self._graph = build_graph(self._llm, self._fleet)
        log.info("AI agent graph compiled with specialists: fleet, command, experiment, topic_link, logs, conversation")

    async def chat(self, message: str, session_id: str, is_voice: bool = False) -> dict:
        """Process one user message through the multi-specialist graph.

        Args:
            message:    The user's plain-text message.
            session_id: Identifies the conversation.
            is_voice:   True if the request came from the voice endpoint.

        Returns:
            Dict with keys:
                ``response``      — The assistant's final text reply.
                ``tool_calls``    — List of tool call dicts.
                ``intent``        — Classified intent category.
                ``voice_summary`` — Short spoken version (only if is_voice).
        """
        DB.upsert_conversation(session_id)
        history = DB.get_conversation_turns(session_id)

        # Build message list from conversation history
        messages = []
        for role, content in history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # Add the current user message
        messages.append(HumanMessage(content=message))

        # Run the graph
        input_state = {
            "messages": messages,
            "intent": "",
            "fleet_context": "",
            "session_id": session_id,
            "tool_calls_made": [],
            "is_voice": is_voice,
            "voice_summary": "",
        }

        result = await self._graph.ainvoke(input_state)

        # Extract final response
        all_messages = result.get("messages", [])
        response = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                response = msg.content
                break

        if not response:
            response = "No response."

        # Strip any remaining think blocks
        response = re.sub(r"<think>[\s\S]*?</think>\s*", "", response).strip()

        # Persist
        DB.save_conversation_turns(session_id, message, response)

        result_dict = {
            "response": response,
            "tool_calls": result.get("tool_calls_made", []),
            "intent": result.get("intent", ""),
        }

        if is_voice:
            result_dict["voice_summary"] = result.get("voice_summary", response)

        return result_dict

    async def chat_stream(self, message: str, session_id: str):
        """Stream the graph execution as events for SSE.

        Yields dicts with ``type`` key:
            - ``intent``      — classified intent
            - ``tool_call``   — tool invocation started
            - ``tool_result`` — tool completed
            - ``token``       — streaming LLM token
            - ``done``        — final response
        """
        DB.upsert_conversation(session_id)
        history = DB.get_conversation_turns(session_id)

        messages = []
        for role, content in history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))

        input_state = {
            "messages": messages,
            "intent": "",
            "fleet_context": "",
            "session_id": session_id,
            "tool_calls_made": [],
            "is_voice": False,
            "voice_summary": "",
        }

        full_response = ""
        intent_sent = False

        # Nodes whose chat-model output should NOT leak to the user's
        # transcript — they are internal LLM calls (e.g. intent classifier,
        # voice-summary generator) that just return control labels.
        SUPPRESSED_NODES = {"classify_intent", "format_response"}

        def _origin_node(ev: dict) -> str:
            md = ev.get("metadata") or {}
            return md.get("langgraph_node") or ""

        async for event in self._graph.astream_events(input_state, version="v2"):
            kind = event.get("event", "")

            # Emit intent after classification
            if kind == "on_chain_end" and not intent_sent:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and "intent" in output and output["intent"]:
                    yield {"type": "intent", "intent": output["intent"]}
                    intent_sent = True

            elif kind == "on_chat_model_stream":
                if _origin_node(event) in SUPPRESSED_NODES:
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield {"type": "token", "content": chunk.content}

            elif kind == "on_tool_start":
                yield {
                    "type": "tool_call",
                    "name": event.get("name", ""),
                    "args": str(event.get("data", {}).get("input", "")),
                }

            elif kind == "on_tool_end":
                yield {"type": "tool_result", "name": event.get("name", "")}

        # Clean up response
        full_response = re.sub(r"<think>[\s\S]*?</think>\s*", "", full_response).strip()
        if not full_response:
            full_response = "No response."

        DB.save_conversation_turns(session_id, message, full_response)

        yield {"type": "done", "response": full_response}
