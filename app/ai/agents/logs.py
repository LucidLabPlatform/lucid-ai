"""Logs specialist — retrieves agent logs and command history."""

import json
from typing import Any

from langchain_core.tools import tool

from app.fleet_client import FleetClient


def build_tools(fleet: FleetClient) -> list:
    """Build log retrieval tools."""

    @tool
    async def get_agent_logs(agent_id: str, limit: int = 50) -> str:
        """Get recent logs for an agent. Returns log entries with timestamps,
        levels, and messages."""
        return _json(await fleet.get_agent_logs(agent_id, limit))

    @tool
    async def get_agent_commands(agent_id: str, limit: int = 50) -> str:
        """Get command history for an agent. Shows sent commands and their results."""
        return _json(await fleet.get_agent_commands(agent_id, limit))

    return [get_agent_logs, get_agent_commands]


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
