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
