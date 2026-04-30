"""Topic link specialist — manages MQTT message routing rules."""

import json
from typing import Any

from langchain_core.tools import tool

from app.fleet_client import FleetClient


def build_tools(fleet: FleetClient) -> list:
    """Build topic link management tools."""

    @tool
    async def list_topic_links() -> str:
        """List all MQTT topic links (rules that forward messages between topics)."""
        return _json(await fleet.list_topic_links())

    @tool
    async def get_topic_link(link_id: str) -> str:
        """Get details and EMQX rule metrics for a topic link — throughput, matched/failed
        message counts, latency. Always call this after list_topic_links when the user
        asks about link health, throughput, or message counts."""
        return _json(await fleet.get_topic_link(link_id))

    @tool
    async def create_topic_link(
        name: str,
        source_topic: str,
        target_topic: str,
        select_clause: str = "*",
        payload_template: str = "",
        qos: int = 0,
    ) -> str:
        """Create a new MQTT topic link that forwards messages from source_topic
        to target_topic."""
        return _json(
            await fleet.create_topic_link(
                name, source_topic, target_topic,
                select_clause, payload_template or None, qos,
            )
        )

    @tool
    async def activate_topic_link(link_id: str) -> str:
        """Activate a topic link so it starts forwarding messages."""
        return _json(await fleet.activate_topic_link(link_id))

    @tool
    async def deactivate_topic_link(link_id: str) -> str:
        """Deactivate a topic link to stop it from forwarding messages."""
        return _json(await fleet.deactivate_topic_link(link_id))

    return [
        list_topic_links,
        get_topic_link,
        create_topic_link,
        activate_topic_link,
        deactivate_topic_link,
    ]


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
