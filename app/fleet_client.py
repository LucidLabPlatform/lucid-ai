import os
from typing import Any

import httpx


class FleetClient:
    def __init__(self, base_url: str | None = None) -> None:
        resolved = base_url or os.environ.get("LUCID_FLEET_CORE_URL") or os.environ["ORCHESTRATOR_URL"]
        self._base_url = resolved.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=35.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> Any:
        response = await self._client.request(
            method=method,
            url=path,
            json=json_body,
            timeout=timeout_s + 5.0,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    async def send_command(
        self,
        *,
        agent_id: str,
        action: str,
        component_id: str | None = None,
        payload: dict | None = None,
        timeout_s: float = 30.0,
    ) -> dict:
        return await self._request(
            "POST",
            "/api/internal/command",
            json_body={
                "agent_id": agent_id,
                "component_id": component_id,
                "action": action,
                "payload": payload or {},
                "wait": True,
                "timeout_s": timeout_s,
            },
            timeout_s=timeout_s,
        )

    async def list_agents(self) -> list[dict]:
        return await self._request("GET", "/api/agents")

    async def get_agent(self, agent_id: str) -> dict:
        return await self._request("GET", f"/api/agents/{agent_id}")

    async def list_experiment_templates(self) -> list[dict]:
        return await self._request("GET", "/api/experiments/templates")

    async def list_experiment_runs(self, status: str | None = None) -> list[dict]:
        path = "/api/experiments/runs"
        if status:
            path = f"{path}?status={status}"
        return await self._request("GET", path)

    async def get_experiment_run(self, run_id: str) -> dict:
        return await self._request("GET", f"/api/experiments/runs/{run_id}")

    async def start_experiment(self, template_id: str, params: dict[str, Any] | None = None) -> dict:
        return await self._request(
            "POST",
            "/api/experiments/run",
            json_body={"template_id": template_id, "params": params or {}},
        )

    async def cancel_experiment_run(self, run_id: str) -> dict:
        return await self._request("DELETE", f"/api/experiments/runs/{run_id}")

    async def approve_experiment(self, run_id: str) -> dict:
        return await self._request("POST", f"/api/experiments/runs/{run_id}/approve")

    # ── Agent logs & commands ────────────────────────────────────────

    async def get_agent_logs(self, agent_id: str, limit: int = 50) -> list[dict]:
        return await self._request("GET", f"/api/agents/{agent_id}/logs?limit={limit}")

    async def get_agent_commands(self, agent_id: str, limit: int = 50) -> list[dict]:
        return await self._request("GET", f"/api/agents/{agent_id}/commands?limit={limit}")

    # ── Direct commands ──────────────────────────────────────────────

    async def send_agent_command(self, agent_id: str, action: str, payload: dict | None = None) -> dict:
        return await self.send_command(
            agent_id=agent_id, action=action, payload=payload, timeout_s=15.0,
        )

    async def send_component_command(
        self, agent_id: str, component_id: str, action: str, payload: dict | None = None
    ) -> dict:
        return await self.send_command(
            agent_id=agent_id, component_id=component_id,
            action=action, payload=payload, timeout_s=15.0,
        )

    async def delete_agent(self, agent_id: str) -> dict:
        return await self._request("DELETE", f"/api/agents/{agent_id}")

    async def get_command_catalog(self, agent_id: str) -> dict:
        return await self._request("GET", f"/api/agents/{agent_id}/command-catalog")

    # ── Topic links ──────────────────────────────────────────────────

    async def list_topic_links(self) -> list[dict]:
        return await self._request("GET", "/api/topic-links")

    async def get_topic_link(self, link_id: str) -> dict:
        return await self._request("GET", f"/api/topic-links/{link_id}")

    async def create_topic_link(
        self, name: str, source_topic: str, target_topic: str,
        select_clause: str = "*", payload_template: str | None = None, qos: int = 0,
    ) -> dict:
        return await self._request(
            "POST",
            "/api/topic-links",
            json_body={
                "name": name,
                "source_topic": source_topic,
                "target_topic": target_topic,
                "select_clause": select_clause,
                "payload_template": payload_template,
                "qos": qos,
            },
        )

    async def activate_topic_link(self, link_id: str) -> dict:
        return await self._request("PUT", f"/api/topic-links/{link_id}/activate")

    async def deactivate_topic_link(self, link_id: str) -> dict:
        return await self._request("PUT", f"/api/topic-links/{link_id}/deactivate")

    async def delete_topic_link(self, link_id: str) -> dict:
        return await self._request("DELETE", f"/api/topic-links/{link_id}")

    # ── Sync state ───────────────────────────────────────────────────

    async def get_sync_state(self) -> dict:
        return await self._request("GET", "/api/sync-state")
