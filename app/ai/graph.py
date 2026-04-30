"""LangGraph StateGraph for the LUCID AI multi-specialist agent system.

Graph topology:
    START → inject_context → classify_intent → [specialist] → format_response → END

Each specialist is a pre-compiled ``create_react_agent`` with a focused set of
tools and a domain-specific system prompt. The graph is compiled once at startup;
per-request, only fleet_context and messages are injected.
"""

import asyncio
import functools
import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.ai.agents import command, experiment, fleet, logs, topic_link
from app.ai.intent import classify_intent
from app.ai.prompts import (
    CLASSIFY_PROMPT,
    COMMAND_SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    EXPERIMENT_SYSTEM_PROMPT,
    FLEET_SYSTEM_PROMPT,
    LOGS_SYSTEM_PROMPT,
    TOPIC_LINK_SYSTEM_PROMPT,
    VOICE_SUMMARY_PROMPT,
)
from app.ai.schema_block import build_schema_block
from app.ai.state import AgentState
from app.fleet_client import FleetClient

log = logging.getLogger(__name__)


# ── Specialist registry ──────────────────────────────────────────────────────

_ALL_SPECIALISTS = {
    "fleet": {
        "prompt": FLEET_SYSTEM_PROMPT,
        "build_tools": fleet.build_tools,
    },
    "command": {
        "prompt": COMMAND_SYSTEM_PROMPT,
        "build_tools": command.build_tools,
    },
    "experiment": {
        "prompt": EXPERIMENT_SYSTEM_PROMPT,
        "build_tools": experiment.build_tools,
    },
    "topic_link": {
        "prompt": TOPIC_LINK_SYSTEM_PROMPT,
        "build_tools": topic_link.build_tools,
    },
    "logs": {
        "prompt": LOGS_SYSTEM_PROMPT,
        "build_tools": logs.build_tools,
    },
}


# Active specialists are selected by the LUCID_AI_TOOLS_PROFILE env var.
# v1: fleet + experiment only (narrow surface for tool-call reliability).
# full: all specialists.
_PROFILE_SPECIALISTS = {
    "v1": ("fleet", "experiment"),
    "full": tuple(_ALL_SPECIALISTS.keys()),
}


def _active_specialists() -> dict:
    profile = os.environ.get("LUCID_AI_TOOLS_PROFILE", "v1").strip().lower()
    names = _PROFILE_SPECIALISTS.get(profile)
    if names is None:
        log.warning(
            "Unknown LUCID_AI_TOOLS_PROFILE=%r; falling back to v1.", profile,
        )
        names = _PROFILE_SPECIALISTS["v1"]
    return {n: _ALL_SPECIALISTS[n] for n in names}


# Backwards-compatible name (kept so external imports/tests don't break).
SPECIALIST_CONFIG = _active_specialists()


class _SafeFormatDict(dict):
    """str.format_map helper: missing keys render as empty strings."""

    def __missing__(self, key):
        return ""


def _wrap_tools_safe(tools: list) -> list:
    """Wrap each tool's coroutine so errors return strings instead of crashing."""
    for t in tools:
        original = t.coroutine

        @functools.wraps(original)
        async def safe_coro(*args, _orig=original, **kwargs):
            try:
                return await _orig(*args, **kwargs)
            except Exception as e:
                log.warning("Tool %s failed: %s", _orig.__name__, e)
                return f"Error: {e}"

        t.coroutine = safe_coro
    return tools


def build_graph(llm: ChatOllama, fleet_client: FleetClient) -> StateGraph:
    """Build and return the compiled LangGraph StateGraph.

    The graph is compiled once at startup. Specialist ``create_react_agent``
    instances are pre-built with their tools bound to the LLM.
    """
    active_specialists = _active_specialists()
    schema_block = build_schema_block()

    try:
        max_iters = int(os.environ.get("LUCID_AI_MAX_ITERATIONS", "6"))
    except ValueError:
        max_iters = 6
    if max_iters < 1:
        max_iters = 6

    log.info(
        "AI graph: profile=%s specialists=%s max_iters=%d",
        os.environ.get("LUCID_AI_TOOLS_PROFILE", "v1"),
        list(active_specialists.keys()),
        max_iters,
    )

    # Pre-build specialist agents
    specialist_agents = {}
    for name, config in active_specialists.items():
        tools = _wrap_tools_safe(config["build_tools"](fleet_client))
        specialist_agents[name] = create_react_agent(model=llm, tools=tools)

    # ── Node functions ───────────────────────────────────────────────

    async def inject_context(state: AgentState) -> dict:
        """Fetch live fleet state and build the context string."""
        try:
            agents = await fleet_client.list_agents()
        except Exception:
            return {"fleet_context": "Fleet unavailable."}

        if not agents:
            return {"fleet_context": "No agents registered."}

        lines = []
        for agent in agents:
            aid = agent["agent_id"]
            status = (agent.get("status") or {}).get("state", "unknown")
            comp_ids = sorted((agent.get("components") or {}).keys())
            comp_str = ", ".join(comp_ids) if comp_ids else "none"
            lines.append(f"- {aid} ({status}) — components: {comp_str}")
        return {"fleet_context": "\n".join(lines)}

    async def classify_intent_node(state: AgentState) -> dict:
        """Classify the user's intent using keyword matching + LLM fallback."""
        # Get the last user message
        user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_msg = msg.content
                break

        intent, confidence, method = await classify_intent(
            user_msg, llm=llm, classify_prompt=CLASSIFY_PROMPT,
        )

        # Under a narrowed profile, route intents we don't have a specialist for
        # to the conversation agent (which can still answer or guide the user).
        if intent != "conversation" and intent not in specialist_agents:
            log.info(
                "Intent %s not in active profile; routing to conversation.",
                intent,
            )
            intent = "conversation"

        log.info(
            '{"event": "intent_classified", "intent": "%s", "confidence": %.2f, "method": "%s"}',
            intent, confidence, method,
        )
        return {"intent": intent}

    def _route_by_intent(state: AgentState) -> str:
        """Conditional edge: route to the specialist node by intent."""
        return state["intent"]

    async def _run_specialist(state: AgentState, name: str) -> dict:
        """Run a pre-compiled specialist agent with focused prompt + fleet context."""
        config = active_specialists[name]
        system_prompt = config["prompt"].format_map(_SafeFormatDict(
            fleet_context=state["fleet_context"],
            schema_block=schema_block,
        ))

        # Build message list: system prompt + conversation history
        messages = [SystemMessage(content=system_prompt)]
        for msg in state["messages"]:
            messages.append(msg)

        agent = specialist_agents[name]
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": messages},
                    config={"recursion_limit": max_iters},
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            return {
                "messages": [AIMessage(content="The request timed out. Please try again.")],
                "tool_calls_made": [],
            }
        except Exception as e:
            log.warning("Specialist %s failed: %s", name, e)
            return {
                "messages": [AIMessage(content="I had trouble processing that. Could you rephrase?")],
                "tool_calls_made": [],
            }

        # Extract tool calls from the agent's messages
        tool_calls = []
        all_messages = result.get("messages", [])
        for msg in all_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": _stringify_args(tc.get("args", {})),
                    })

        # Get the final response message
        response_msgs = [m for m in all_messages if isinstance(m, AIMessage) and m.content]
        final_msg = response_msgs[-1] if response_msgs else AIMessage(content="No response.")

        log.info(
            '{"event": "specialist_done", "specialist": "%s", "tool_count": %d}',
            name, len(tool_calls),
        )

        return {
            "messages": [final_msg],
            "tool_calls_made": tool_calls,
        }

    async def conversation_agent_node(state: AgentState) -> dict:
        """Direct LLM call for conversational responses (no tools)."""
        system_prompt = CONVERSATION_SYSTEM_PROMPT.format_map(_SafeFormatDict(
            fleet_context=state["fleet_context"],
            schema_block=schema_block,
        ))
        messages = [SystemMessage(content=system_prompt)]
        for msg in state["messages"]:
            messages.append(msg)

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=30.0,
            )
        except Exception as e:
            log.warning("Conversation agent failed: %s", e)
            response = AIMessage(content="I had trouble responding. Could you rephrase?")

        content = response.content if hasattr(response, "content") else str(response)
        return {
            "messages": [AIMessage(content=content)],
            "tool_calls_made": [],
        }

    async def format_response(state: AgentState) -> dict:
        """Strip think blocks and optionally generate voice summary."""
        last_msg = state["messages"][-1]
        content = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)

        # Strip Qwen3 <think> blocks
        content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()

        updates: dict[str, Any] = {
            "messages": [AIMessage(content=content)],
        }

        # Generate voice summary if this is a voice request
        if state.get("is_voice") and content:
            try:
                summary_prompt = VOICE_SUMMARY_PROMPT.format(response=content)
                summary_resp = await llm.ainvoke([
                    ("system", summary_prompt),
                ])
                voice_text = summary_resp.content.strip()
                voice_text = re.sub(r"<think>[\s\S]*?</think>\s*", "", voice_text).strip()
                updates["voice_summary"] = voice_text
            except Exception as e:
                log.warning("Voice summary generation failed: %s", e)
                # Fall back to the first sentence
                updates["voice_summary"] = content.split(".")[0] + "."

        return updates

    # ── Build the graph ──────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("inject_context", inject_context)
    graph.add_node("classify_intent", classify_intent_node)

    # Add specialist nodes
    for name in active_specialists:
        # Use a factory to capture the name correctly in the closure
        def make_node(specialist_name: str):
            async def node_fn(state: AgentState) -> dict:
                return await _run_specialist(state, specialist_name)
            node_fn.__name__ = f"{specialist_name}_agent"
            return node_fn

        graph.add_node(f"{name}_agent", make_node(name))

    graph.add_node("conversation_agent", conversation_agent_node)
    graph.add_node("format_response", format_response)

    # Edges
    graph.set_entry_point("inject_context")
    graph.add_edge("inject_context", "classify_intent")

    # Conditional routing by intent
    route_map = {name: f"{name}_agent" for name in active_specialists}
    route_map["conversation"] = "conversation_agent"
    graph.add_conditional_edges("classify_intent", _route_by_intent, route_map)

    # All specialists → format_response → END
    for name in active_specialists:
        graph.add_edge(f"{name}_agent", "format_response")
    graph.add_edge("conversation_agent", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


def _stringify_args(args: Any) -> str:
    if isinstance(args, dict):
        if "task" in args and len(args) == 1:
            return str(args["task"])
        return json.dumps(args, sort_keys=True, default=str)
    return str(args)
