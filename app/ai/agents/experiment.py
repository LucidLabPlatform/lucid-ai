"""Experiment specialist — manages experiment templates and runs."""

import json
from typing import Any

from langchain_core.tools import tool

from app.fleet_client import FleetClient


def build_tools(fleet: FleetClient) -> list:
    """Build experiment management tools."""

    @tool
    async def list_experiment_templates() -> str:
        """List available experiment templates with ids, names, versions, and tags.
        Call configure_experiment(id) next to get the parameter schema before starting."""
        templates = await fleet.list_experiment_templates()
        summary = [
            {
                "id": t["id"],
                "name": t["name"],
                "version": t["version"],
                "description": t.get("description", ""),
                "tags": t.get("tags", []),
            }
            for t in templates
        ]
        return _json(summary)

    @tool
    async def configure_experiment(template_id: str) -> str:
        """Get the parameter schema for a specific experiment template.
        Call this after list_experiment_templates and before start_experiment
        to know which parameters are required vs optional and their types."""
        template = await fleet.get_experiment_template(template_id)
        return _json({
            "id": template["id"],
            "name": template["name"],
            "parameters_schema": template.get("parameters_schema", {}),
        })

    @tool
    async def start_experiment(
        template_id: str, params: dict[str, Any] | None = None
    ) -> str:
        """Start an experiment run. ALWAYS call configure_experiment(template_id) first
        to discover required parameters — never call this without knowing what params
        are needed."""
        result = await fleet.start_experiment(template_id, params=_coerce(params))
        return _json(result)

    @tool
    async def start_default_experiment(template_id: str = "") -> str:
        """Start an experiment using sensible defaults for every parameter.
        Use this when the user says "start an experiment" without specifying
        which template or which parameters. With no template_id, picks the
        canonical foraging experiment.

        How it works: fetches the template's parameters_schema and submits each
        parameter's `default` value automatically. The user does not need to
        provide any arguments.
        """
        # Pick the template
        chosen = (template_id or "").strip()
        if not chosen:
            try:
                templates = await fleet.list_experiment_templates()
            except Exception as e:
                return _json({"ok": False, "error": f"Could not list templates: {e}"})
            ids = [t.get("id", "") for t in templates]
            preferred = ("foraging-experiment-v2", "foraging-experiment", "foraging-run-v2", "foraging-run")
            chosen = next((p for p in preferred if p in ids), ids[0] if ids else "")
            if not chosen:
                return _json({"ok": False, "error": "No experiment templates available."})

        # Resolve schema → defaults
        try:
            template = await fleet.get_experiment_template(chosen)
        except Exception as e:
            return _json({"ok": False, "error": f"Template {chosen!r} not found: {e}"})

        schema = template.get("parameters_schema") or {}
        params: dict[str, Any] = {}
        for name, spec in schema.items():
            if not isinstance(spec, dict):
                continue
            if "default" in spec and spec["default"] is not None:
                params[name] = spec["default"]
        # Start it
        result = await fleet.start_experiment(chosen, params=params)
        return _json({
            "ok": True,
            "template_id": chosen,
            "template_name": template.get("name", chosen),
            "params_used": params,
            "run": result,
        })

    @tool
    async def list_experiment_runs(status: str = "") -> str:
        """List experiment runs. Optionally filter by status such as pending,
        running, completed, failed, or cancelled."""
        runs = await fleet.list_experiment_runs(status=status or None)
        summary = [
            {
                "id": r["id"],
                "template_id": r["template_id"],
                "status": r["status"],
                "created_at": r.get("created_at"),
                "started_at": r.get("started_at"),
                "ended_at": r.get("ended_at"),
            }
            for r in runs
        ]
        return _json(summary)

    @tool
    async def get_experiment_run(run_id: str) -> str:
        """Get one experiment run and its recorded steps by exact run_id."""
        return _json(await fleet.get_experiment_run(run_id))

    @tool
    async def approve_experiment_step(run_id: str) -> str:
        """Approve a pending approval step in an experiment run.
        Use when the researcher confirms they are ready to proceed."""
        return _json(await fleet.approve_experiment(run_id))

    return [
        list_experiment_templates,
        configure_experiment,
        start_experiment,
        start_default_experiment,
        list_experiment_runs,
        get_experiment_run,
        approve_experiment_step,
    ]


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
