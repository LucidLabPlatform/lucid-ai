"""Builds the LUCID MQTT contract block injected into specialist system prompts.

Parses ``topics.txt`` into a compact reference of agent + component command
actions, plus the canonical topic patterns. The block is regenerated at AI
service startup; it is small (well under 2000 tokens) so it lives directly in
every prompt rather than in a vector store.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)


_DEFAULT_TOPICS_LOCATIONS = (
    Path("/app/topics.txt"),                                # in-container
    Path(__file__).resolve().parents[3] / "topics.txt",     # repo: lucid-ai/../topics.txt
    Path(__file__).resolve().parents[4] / "topics.txt",     # workspace root
)


def _resolve_topics_path() -> Path | None:
    override = os.environ.get("LUCID_AI_TOPICS_PATH")
    if override:
        p = Path(override)
        return p if p.exists() else None
    for p in _DEFAULT_TOPICS_LOCATIONS:
        if p.exists():
            return p
    return None


def _parse_topics(text: str) -> dict:
    """Return {agent_cmds: [...], components: {comp_name: [cmds]}}."""
    agent_cmds: list[str] = []
    components: dict[str, list[str]] = {}

    # Agent-level commands
    for m in re.finditer(r"^lucid/agents/<agent_id>/cmd/([\w/_-]+)\s*$", text, re.M):
        action = m.group(1).strip()
        if action:
            agent_cmds.append(action)

    # Component-level commands grouped by component type
    current_comp: str | None = None
    in_publish = False
    for raw in text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^COMPONENT:\s*(\S+)", line)
        if m:
            current_comp = m.group(1).strip()
            components.setdefault(current_comp, [])
            in_publish = False
            continue
        if line.startswith("AGENT") or re.match(r"^COMPONENT:", line):
            in_publish = False
        if "Publish (you send" in line:
            in_publish = True
            continue
        if in_publish and current_comp:
            cm = re.search(
                r"components/" + re.escape(current_comp) + r"/cmd/([\w/_-]+)",
                line,
            )
            if cm:
                components[current_comp].append(cm.group(1).strip())

    return {
        "agent_cmds": sorted(set(agent_cmds)),
        "components": {k: sorted(set(v)) for k, v in components.items() if v},
    }


def render_schema_block(parsed: dict) -> str:
    """Render the parsed topics into a compact text block."""
    lines: list[str] = []
    lines.append("LUCID MQTT contract — use these exact topic patterns and action names. Never invent topics or actions.")
    lines.append("")
    lines.append("Topic patterns:")
    lines.append("  lucid/agents/{agent_id}/{retained|stream|cmd/{action}|evt/{action}/result}")
    lines.append("  lucid/agents/{agent_id}/components/{component_id}/{...same subtree}")
    lines.append("Retained: metadata, status, state, cfg, cfg/logging, cfg/telemetry, schema.")
    lines.append("Streams: logs, telemetry/{metric}.")
    lines.append("All command payloads require {request_id: <uuid>}.")
    lines.append("")
    lines.append("Agent commands (cmd/<action>):")
    if parsed.get("agent_cmds"):
        lines.append("  " + ", ".join(parsed["agent_cmds"]))
    else:
        lines.append("  (none parsed)")
    lines.append("")
    lines.append("Component commands by component type:")
    components = parsed.get("components", {})
    if components:
        for comp, cmds in components.items():
            lines.append(f"  {comp}: {', '.join(cmds) if cmds else '(none)'}")
    else:
        lines.append("  (none parsed)")
    return "\n".join(lines)


def build_schema_block() -> str:
    """Build the full schema block from topics.txt; falls back to a minimal stub."""
    path = _resolve_topics_path()
    if path is None:
        log.warning("topics.txt not found in any expected location; using minimal schema stub.")
        return _MINIMAL_STUB
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return _MINIMAL_STUB
    parsed = _parse_topics(text)
    block = render_schema_block(parsed)
    log.info(
        "Built schema block from %s — %d agent cmds, %d component types",
        path, len(parsed["agent_cmds"]), len(parsed["components"]),
    )
    return block


_MINIMAL_STUB = """LUCID MQTT contract — see topics.txt for full reference.
Topic patterns:
  lucid/agents/{agent_id}/{retained|stream|cmd/{action}|evt/{action}/result}
  lucid/agents/{agent_id}/components/{component_id}/{...same subtree}
All command payloads require {request_id: <uuid>}.
"""
