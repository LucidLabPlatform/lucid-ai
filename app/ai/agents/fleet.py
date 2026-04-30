"""Fleet specialist — queries agent/component status."""

import json
from typing import Any

from langchain_core.tools import tool

from app.fleet_client import FleetClient


def build_tools(fleet: FleetClient) -> list:
    """Build fleet query tools."""

    @tool
    async def list_agents() -> str:
        """List known agents with their current status and component ids."""
        agents = await fleet.list_agents()
        summary = [
            {
                "agent_id": a["agent_id"],
                "status": (a.get("status") or {}).get("state"),
                "component_ids": sorted((a.get("components") or {}).keys()),
            }
            for a in agents
        ]
        return _json(summary)

    @tool
    async def get_agent(agent_id: str) -> str:
        """Get full state for a specific agent by exact agent_id."""
        return _json(await fleet.get_agent(agent_id))

    @tool
    async def get_sync_state() -> str:
        """Get the current sync state of all managed domains."""
        return _json(await fleet.get_sync_state())

    @tool
    async def get_fleet_summary() -> str:
        """Get an aggregated snapshot of the fleet: counts of online/offline agents,
        component totals, and any in-progress experiment runs. Use this when the
        user asks for an overview rather than a specific agent."""
        agents = await fleet.list_agents()
        online = 0
        offline = 0
        component_total = 0
        for a in agents:
            state = (a.get("status") or {}).get("state", "").lower()
            if state in ("online", "running", "ready"):
                online += 1
            else:
                offline += 1
            component_total += len((a.get("components") or {}).keys())

        running_runs = 0
        try:
            running = await fleet.list_experiment_runs(status="running")
            running_runs = len(running) if isinstance(running, list) else 0
        except Exception:
            running_runs = -1  # signal: could not fetch

        return _json({
            "agents_total": len(agents),
            "agents_online": online,
            "agents_offline": offline,
            "components_total": component_total,
            "experiment_runs_in_progress": running_runs,
            "agent_ids": sorted([a["agent_id"] for a in agents]),
        })

    return [list_agents, get_agent, get_sync_state, get_fleet_summary]


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
