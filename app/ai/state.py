"""Shared state schema for the LUCID AI LangGraph pipeline."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State that flows through the LangGraph StateGraph.

    Fields:
        messages:        Conversation messages (uses add_messages reducer).
        intent:          Classified intent category for routing.
        fleet_context:   Live agent/component IDs injected into specialist prompts.
        session_id:      Conversation session identifier.
        tool_calls_made: Tool calls accumulated during the specialist's ReAct loop.
        is_voice:        True when the request originated from the voice endpoint.
        voice_summary:   Short TTS-friendly summary (only generated when is_voice).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    fleet_context: str
    session_id: str
    tool_calls_made: list[dict]
    is_voice: bool
    voice_summary: str
