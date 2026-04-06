"""AI Workflow Supervisor Agent for LUCID Central Command.

This module implements the LangGraph ReAct agent that powers the ``/ai`` chat
interface.  The agent acts as a **supervisor** that:

1. Loads the conversation history from Postgres for the given ``session_id``.
2. Queries ``component_metadata`` for online components with
   ``capabilities.type == "ai_specialist"``.
3. Dynamically builds one LangChain ``@tool`` per specialist.  Each tool
   publishes a ``cmd/task`` MQTT command to the specialist's component and
   awaits the ``evt/task/result`` response via ``RequestResponseManager``.
4. Runs the ReAct loop (Reason + Act) using Ollama as the LLM backend.
5. Persists the user+assistant exchange to Postgres.

Environment variables:
    OLLAMA_MODEL        Ollama model tag (default: ``"llama3.1:8b"``).
    OLLAMA_BASE_URL     Ollama HTTP base URL (default: ``"http://ollama:11434"``).
    AI_MAX_ITERATIONS   Maximum ReAct loop iterations (default: ``15``).
    AI_CHAT_TIMEOUT     Overall timeout in seconds (default: ``60.0``).

Key classes:
    AIWorkflowAgent — instantiated once at startup; stateless between calls.
"""
import asyncio
import json
import os
from typing import Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app import db as DB
from app.ai.prompts import SUPERVISOR_SYSTEM_PROMPT


class AIWorkflowAgent:
    """LangGraph ReAct supervisor agent that routes tasks to LUCID specialists.

    Attributes:
        _fleet:  HTTP client for `lucid-orchestrator` command dispatch.
        _llm:    Configured ``ChatOllama`` language model instance.
    """

    def __init__(self, fleet):
        """
        Args:
            fleet: HTTP client that publishes commands through ``lucid-orchestrator``.
        """
        self._fleet = fleet
        self._llm = ChatOllama(
            model=os.environ["OLLAMA_MODEL"],
            temperature=0,
            num_ctx=32768,
            base_url=os.environ["OLLAMA_BASE_URL"],
        )

    async def chat(self, message: str, session_id: str) -> dict:
        """Process one user message through the ReAct agent and return the response.

        Steps:
            1. Upsert conversation record in Postgres.
            2. Load conversation history (provides the agent multi-turn context).
            3. Discover available specialists from ``component_metadata``.
            4. Build specialist tools and construct the ReAct agent.
            5. Run ``agent.ainvoke`` with a configurable timeout.
            6. Extract the final assistant message and any tool calls made.
            7. Persist both turns to Postgres.

        Args:
            message:    The user's plain-text message.
            session_id: Identifies the conversation; can be any stable string.

        Returns:
            Dict with:
                ``response``   — The assistant's final text reply.
                ``tool_calls`` — List of ``{"specialist": str, "task": str}`` dicts.

        Raises:
            asyncio.TimeoutError: If the agent exceeds ``AI_CHAT_TIMEOUT`` seconds.
        """
        DB.upsert_conversation(session_id)
        history = DB.get_conversation_turns(session_id)
        specialists = DB.get_available_specialists()
        tools = self._build_core_tools() + self._build_specialist_tools(specialists)
        system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
            specialist_list="\n".join(
                f"- {s['component_id']}: {s['description']}" for s in specialists
            ) or "No specialist components currently online."
        )

        # A new agent is created per call so the MemorySaver checkpoint reflects
        # the current conversation history injected via the ``messages`` list.
        agent = create_react_agent(
            model=self._llm,
            tools=tools,
            checkpointer=MemorySaver(),
        )

        # Build the message list: system prompt → history → current user message
        messages = [("system", system_prompt)]
        for role, content in history:
            messages.append((role, content))
        messages.append(("user", message))

        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": messages},
                    config={
                        "configurable": {"thread_id": session_id},
                        "recursion_limit": int(os.environ.get("AI_MAX_ITERATIONS", "15")),
                    },
                ),
                timeout=float(os.environ.get("AI_CHAT_TIMEOUT", "60")),
            )
        except asyncio.TimeoutError:
            raise

        all_messages = result.get("messages", [])
        response = all_messages[-1].content if all_messages else "No response"

        # Extract tool calls made during the ReAct loop for the UI to display
        tool_calls = []
        for msg in all_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": self._stringify_tool_args(tc.get("args", {})),
                    })

        DB.save_conversation_turns(session_id, message, response)
        return {"response": response, "tool_calls": tool_calls}

    def _build_core_tools(self) -> list:
        """Create first-party tools backed by lucid-orchestrator APIs.

        These make the AI useful even when no specialist components are online.
        """

        @tool
        async def list_agents() -> str:
            """List known agents with their current status and component ids."""
            agents = await self._fleet.list_agents()
            summary = [
                {
                    "agent_id": agent["agent_id"],
                    "status": (agent.get("status") or {}).get("state"),
                    "component_ids": sorted((agent.get("components") or {}).keys()),
                }
                for agent in agents
            ]
            return self._json_output(summary)

        @tool
        async def get_agent(agent_id: str) -> str:
            """Get full state for a specific agent by exact agent_id."""
            return self._json_output(await self._fleet.get_agent(agent_id))

        @tool
        async def list_experiment_templates() -> str:
            """List available experiment templates with ids, names, versions, and tags."""
            templates = await self._fleet.list_experiment_templates()
            summary = [
                {
                    "id": template["id"],
                    "name": template["name"],
                    "version": template["version"],
                    "description": template.get("description", ""),
                    "tags": template.get("tags", []),
                }
                for template in templates
            ]
            return self._json_output(summary)

        @tool
        async def list_experiment_runs(status: str = "") -> str:
            """List experiment runs. Optionally filter by status such as pending, running, completed, failed, or cancelled."""
            runs = await self._fleet.list_experiment_runs(status=status or None)
            summary = [
                {
                    "id": run["id"],
                    "template_id": run["template_id"],
                    "status": run["status"],
                    "created_at": run.get("created_at"),
                    "started_at": run.get("started_at"),
                    "ended_at": run.get("ended_at"),
                }
                for run in runs
            ]
            return self._json_output(summary)

        @tool
        async def get_experiment_run(run_id: str) -> str:
            """Get one experiment run and its recorded steps by exact run_id."""
            return self._json_output(await self._fleet.get_experiment_run(run_id))

        @tool
        async def start_experiment(template_id: str, params: dict[str, Any] | None = None) -> str:
            """Start an experiment run from an exact template_id and optional params object."""
            result = await self._fleet.start_experiment(template_id, params=params or {})
            return self._json_output(result)

        @tool
        async def cancel_experiment_run(run_id: str) -> str:
            """Cancel a running or pending experiment by exact run_id."""
            result = await self._fleet.cancel_experiment_run(run_id)
            return self._json_output(result)

        @tool
        async def approve_experiment_step(run_id: str) -> str:
            """Approve a pending approval step in an experiment run.
            Use when the researcher confirms they are ready to proceed
            (e.g., pucks have been placed in the arena)."""
            result = await self._fleet.approve_experiment(run_id)
            return self._json_output(result)

        return [
            list_agents,
            get_agent,
            list_experiment_templates,
            list_experiment_runs,
            get_experiment_run,
            start_experiment,
            cancel_experiment_run,
            approve_experiment_step,
        ]

    def _build_specialist_tools(self, specialists: list[dict]) -> list:
        """Dynamically create one LangChain tool per online specialist component.

        Each tool publishes a ``cmd/task`` MQTT command to the specialist's
        ``lucid/agents/{agent_id}/components/{component_id}/cmd/task`` topic and
        awaits the ``evt/task/result`` response via the RRM.

        Default values for ``_aid`` and ``_cid`` in the closure capture the
        correct agent/component IDs for each specialist; without them the closure
        would capture the loop variable reference (Python gotcha).

        Args:
            specialists: List of dicts with ``agent_id``, ``component_id``,
                         and ``description`` fields from ``DB.get_available_specialists()``.

        Returns:
            List of LangChain ``@tool``-decorated async functions.
        """
        tools = []
        for s in specialists:
            agent_id = s["agent_id"]
            component_id = s["component_id"]
            description = s.get("description") or f"AI specialist: {component_id}"

            @tool(name=component_id, description=description)
            async def call_specialist(task: str, _aid=agent_id, _cid=component_id) -> str:
                """Call a specialist component with a specific task description."""
                try:
                    response = await self._fleet.send_command(
                        agent_id=_aid,
                        component_id=_cid,
                        action="task",
                        payload={"task": task},
                        timeout_s=30.0,
                    )
                    result = response.get("result")
                    if isinstance(result, dict):
                        return result.get("result", result.get("error", json.dumps(result)))
                    if result is None:
                        return "no response"
                    return str(result)
                except asyncio.TimeoutError:
                    return f"{_cid} timed out — is the specialist running?"
                except Exception as e:
                    return f"{_cid} error: {e}"

            tools.append(call_specialist)
        return tools

    @staticmethod
    def _json_output(value: Any) -> str:
        return json.dumps(value, indent=2, sort_keys=True, default=str)

    @staticmethod
    def _stringify_tool_args(args: Any) -> str:
        if isinstance(args, dict):
            if "task" in args and len(args) == 1:
                return str(args["task"])
            return json.dumps(args, sort_keys=True, default=str)
        return str(args)
