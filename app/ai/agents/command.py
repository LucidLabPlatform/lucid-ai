"""Command specialist — sends commands to agents and components."""

import json
from typing import Any

from langchain_core.tools import tool

from app.fleet_client import FleetClient


def build_tools(fleet: FleetClient) -> list:
    """Build command dispatch tools."""

    @tool
    async def get_command_catalog(agent_id: str) -> str:
        """Get the full catalog of available commands for an agent and its components.
        Returns action names, expected payload templates, and categories.
        ALWAYS call this before send_agent_command or send_component_command."""
        return _json(await fleet.get_command_catalog(agent_id))

    @tool
    async def send_agent_command(agent_id: str, action: str, payload: dict | None = None) -> str:
        """Send a command to an agent. ALWAYS call get_command_catalog first."""
        return _json(
            await fleet.send_agent_command(agent_id, action, _coerce(payload))
        )

    @tool
    async def send_component_command(
        agent_id: str, component_id: str, action: str, payload: dict | None = None
    ) -> str:
        """Send a command to a component. ALWAYS call get_command_catalog first."""
        return _json(
            await fleet.send_component_command(
                agent_id, component_id, action, _coerce(payload)
            )
        )

    @tool
    async def send_batch_command(
        action: str, targets: list[dict], payload: dict | None = None
    ) -> str:
        """Send the same command to multiple agents/components in parallel.
        Each target needs agent_id and optionally component_id.
        ALWAYS call get_command_catalog first for at least one target agent."""
        return _json(
            await fleet.send_batch_command(action, targets, _coerce(payload))
        )

    return [get_command_catalog, send_agent_command, send_component_command, send_batch_command]


def _coerce(payload: Any) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        cleaned = payload.strip()
        if not cleaned:
            return {}
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
