"""Unit tests for the schema_block builder."""

from __future__ import annotations

import textwrap

from app.ai.schema_block import _parse_topics, render_schema_block


_FIXTURE = textwrap.dedent(
    """\
    LUCID MQTT Topic Reference (current)

    AGENT - SUBSCRIBE
    lucid/agents/<agent_id>/metadata
    lucid/agents/<agent_id>/status

    AGENT - PUBLISH (you publish, agent receives)
    lucid/agents/<agent_id>/cmd/ping
    lucid/agents/<agent_id>/cmd/restart
    lucid/agents/<agent_id>/cmd/components/install

    COMPONENT: led_strip
    Subscribe (component publishes)
    lucid/agents/<agent_id>/components/led_strip/metadata

    Publish (you send, component receives)
    lucid/agents/<agent_id>/components/led_strip/cmd/set-color
    lucid/agents/<agent_id>/components/led_strip/cmd/clear

    COMPONENT: fixture_cpu
    Subscribe (component publishes)
    lucid/agents/<agent_id>/components/fixture_cpu/metadata

    Publish (you send, component receives)
    lucid/agents/<agent_id>/components/fixture_cpu/cmd/reset
    """,
)


def test_parse_topics_extracts_agent_commands():
    parsed = _parse_topics(_FIXTURE)
    assert "ping" in parsed["agent_cmds"]
    assert "restart" in parsed["agent_cmds"]
    assert "components/install" in parsed["agent_cmds"]


def test_parse_topics_extracts_component_commands():
    parsed = _parse_topics(_FIXTURE)
    assert parsed["components"]["led_strip"] == ["clear", "set-color"]
    assert parsed["components"]["fixture_cpu"] == ["reset"]


def test_render_block_includes_required_sections():
    parsed = _parse_topics(_FIXTURE)
    block = render_schema_block(parsed)
    assert "lucid/agents/{agent_id}" in block
    assert "request_id" in block
    assert "ping" in block
    assert "led_strip" in block
    assert "set-color" in block


def test_render_block_token_budget():
    parsed = _parse_topics(_FIXTURE)
    block = render_schema_block(parsed)
    # Cheap proxy for "fits in 2000 tokens": well under 8000 chars.
    assert len(block) < 8000
