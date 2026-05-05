#!/usr/bin/env python3
"""LUCID AI accuracy test for the multi-specialist agent system.

Tests two dimensions:
1. **Intent classification** — every query is verified to route to the correct
   specialist (fleet, command, experiment, topic_link, logs, conversation).
2. **Tool selection** — for queries that require tool calls, verify the right
   tools are invoked in the right order.

The active specialist profile is read from LUCID_AI_TOOLS_PROFILE on the AI
service. Under "v1" only fleet + experiment specialists are active; queries
in other domains are expected to route to conversation. Under "full" all
specialists are active.

Run while the stack is up:

    ssh failsafe@10.205.10.16 \\
      'cd ~/lucid-central-command/lucid-ai && \\
       python tests/ai_accuracy_test.py --url http://localhost:5000'

Options:
    --url      Base URL (default: $AI_URL or http://localhost:5000)
    --profile  v1 (default) or full — controls which categories are scored

Exit code:
    0  accuracy >= threshold
    1  accuracy <  threshold
"""

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field

import httpx

# ── ANSI colours ─────────────────────────────────────────────────────────────
_NO_COLOUR = not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return text if _NO_COLOUR else f"{code}{text}\033[0m"


GREEN  = lambda t: _c("\033[32m", t)  # noqa: E731
RED    = lambda t: _c("\033[31m", t)  # noqa: E731
YELLOW = lambda t: _c("\033[33m", t)  # noqa: E731
CYAN   = lambda t: _c("\033[36m", t)  # noqa: E731
GREY   = lambda t: _c("\033[90m", t)  # noqa: E731
BOLD   = lambda t: _c("\033[1m",  t)  # noqa: E731

PASS_THRESHOLD = 0.90  # raised from 0.80 — we expect near-perfect routing now

# Profiles: which intents are served by their own specialist (vs conversation).
PROFILE_ACTIVE_INTENTS = {
    "v1":   {"fleet", "experiment", "conversation"},
    "full": {"fleet", "command", "experiment", "topic_link", "logs", "conversation"},
}


# ── Test case definition ──────────────────────────────────────────────────────
@dataclass
class Case:
    """One test query."""
    query: str
    expected_intent: str
    required_tools: list[str] = field(default_factory=list)
    # If True, required_tools must appear in that exact order.
    ordered: bool = False
    description: str = ""
    # If True, the case passes even when no tools are called — answered from context.
    allow_no_tools: bool = False
    # Filled in at runtime.
    result_intent: str = ""
    result_tools: list[str] = field(default_factory=list)
    result_response: str = ""
    passed: bool | None = None
    fail_reason: str = ""
    error: str = ""
    # If True, this case only runs when the matching specialist is active.
    skip_in_v1: bool = False


# ── Fleet probe ───────────────────────────────────────────────────────────────
def probe_fleet(client: httpx.Client, fleet_url: str) -> list[dict]:
    """Return agent list from the orchestrator's /api/agents endpoint."""
    try:
        resp = client.get(f"{fleet_url}/api/agents", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("agents", [])
    except Exception as exc:
        print(YELLOW(f"  Warning: could not probe fleet ({exc}) — using placeholder IDs"))
        return []


# ── Case builder ──────────────────────────────────────────────────────────────
def build_cases(agents: list[dict]) -> list[Case]:
    """Build the test cases, substituting real agent IDs where available."""

    agent_ids = [a["agent_id"] for a in agents]

    def pick_agent(keyword: str, fallback: str) -> str:
        for a in agents:
            comps = list((a.get("components") or {}).keys())
            if any(keyword in c for c in comps):
                return a["agent_id"]
        for a in agents:
            if keyword in a["agent_id"]:
                return a["agent_id"]
        return agent_ids[0] if agent_ids else fallback

    first_agent  = agent_ids[0] if agent_ids else "robot-01"
    led_agent    = pick_agent("led",     "led-agent-01")
    robot_agent  = pick_agent("ros",     "robot-agent-01")
    camera_agent = pick_agent("camera",  first_agent)

    return [
        # ── Fleet (5) — covered by v1 profile ─────────────────────────────────
        Case(
            query="What devices are online?",
            expected_intent="fleet",
            required_tools=[],  # answerable from fleet context
            allow_no_tools=True,
            description="fleet status from context",
        ),
        Case(
            query=f"Show me the full state of {first_agent}",
            expected_intent="fleet",
            required_tools=["get_agent"],
            description="single agent detail",
        ),
        Case(
            query="Give me a fleet summary",
            expected_intent="fleet",
            required_tools=["get_fleet_summary"],
            allow_no_tools=True,  # may answer from injected fleet_context
            description="aggregated fleet snapshot",
        ),
        Case(
            query=f"What components does {first_agent} have?",
            expected_intent="fleet",
            required_tools=[],
            allow_no_tools=True,
            description="component enumeration from context",
        ),
        Case(
            query="Are any agents offline?",
            expected_intent="fleet",
            required_tools=[],
            allow_no_tools=True,
            description="offline check from context",
        ),

        # ── Experiment (5) — covered by v1 profile ────────────────────────────
        Case(
            query="What experiment templates are available?",
            expected_intent="experiment",
            required_tools=["list_experiment_templates"],
            description="template discovery",
        ),
        Case(
            query="Show me the most recent experiment run",
            expected_intent="experiment",
            required_tools=["list_experiment_runs"],
            description="run history",
        ),
        Case(
            query="What's the status of the last experiment?",
            expected_intent="experiment",
            required_tools=["list_experiment_runs"],
            description="run status (rephrased)",
        ),
        Case(
            query="List all running experiments",
            expected_intent="experiment",
            required_tools=["list_experiment_runs"],
            description="filter runs by status",
        ),
        Case(
            query="What parameters does the foraging experiment need?",
            expected_intent="experiment",
            required_tools=["list_experiment_templates", "configure_experiment"],
            ordered=True,
            description="template parameter discovery",
        ),

        # ── Conversation (3) — covered by v1 profile ──────────────────────────
        Case(
            query="What is LUCID?",
            expected_intent="conversation",
            required_tools=[],
            allow_no_tools=True,
            description="LUCID explanation",
        ),
        Case(
            query="Hello, what can you do?",
            expected_intent="conversation",
            required_tools=[],
            allow_no_tools=True,
            description="capability question",
        ),
        Case(
            query="What's the weather like?",
            expected_intent="conversation",
            required_tools=[],
            allow_no_tools=True,
            description="off-topic — should refuse politely",
        ),

        # ── Command (4) — only in full profile ────────────────────────────────
        Case(
            query=f"Set the LED strip on {led_agent} to red",
            expected_intent="command",
            required_tools=["get_command_catalog", "send_component_command"],
            ordered=True,
            description="command: catalog → send (colour)",
            skip_in_v1=True,
        ),
        Case(
            query=f"Turn off the LED strip on {led_agent}",
            expected_intent="command",
            required_tools=["get_command_catalog", "send_component_command"],
            ordered=True,
            description="command: catalog → send (off)",
            skip_in_v1=True,
        ),
        Case(
            query=f"What commands can I send to {first_agent}?",
            expected_intent="command",
            required_tools=["get_command_catalog"],
            description="command discovery only",
            skip_in_v1=True,
        ),
        Case(
            query="Ping all agents",
            expected_intent="command",
            required_tools=["get_command_catalog", "send_batch_command"],
            ordered=True,
            description="batch command across agents",
            skip_in_v1=True,
        ),

        # ── Topic links (3) — only in full profile ────────────────────────────
        Case(
            query="Show me all topic links",
            expected_intent="topic_link",
            required_tools=["list_topic_links"],
            description="list routing rules",
            skip_in_v1=True,
        ),
        Case(
            query="What's the throughput on the perception link?",
            expected_intent="topic_link",
            required_tools=["list_topic_links", "get_topic_link"],
            ordered=True,
            description="list → get metrics",
            skip_in_v1=True,
        ),
        Case(
            query=f"Link robot perception on {robot_agent} to the LED strip on {led_agent}",
            expected_intent="topic_link",
            required_tools=["create_topic_link"],
            description="create EMQX republish rule",
            skip_in_v1=True,
        ),

        # ── Logs (3) — only in full profile ───────────────────────────────────
        Case(
            query=f"Show me recent logs from {first_agent}",
            expected_intent="logs",
            required_tools=["get_agent_logs"],
            description="agent log retrieval",
            skip_in_v1=True,
        ),
        Case(
            query=f"What commands were sent to {first_agent} recently?",
            expected_intent="logs",
            required_tools=["get_agent_commands"],
            description="command history",
            skip_in_v1=True,
        ),
        Case(
            query=f"Show me the command history for {camera_agent}",
            expected_intent="logs",
            required_tools=["get_agent_commands"],
            description="command history (rephrased)",
            skip_in_v1=True,
        ),
    ]


# ── Query runner ──────────────────────────────────────────────────────────────
def run_query(client: httpx.Client, base_url: str, query: str) -> tuple[str, list[str], str]:
    """POST to /api/ai/chat; return (intent, tool_names_in_order, response_text)."""
    resp = client.post(
        f"{base_url}/api/ai/chat",
        json={"message": query, "session_id": str(uuid.uuid4())},
        timeout=180.0,
    )
    resp.raise_for_status()
    data = resp.json()
    intent = data.get("intent", "")
    tool_names = [tc.get("name", "") for tc in data.get("tool_calls", [])]
    response = data.get("response", "")
    return intent, tool_names, response


# ── Pass/fail logic ───────────────────────────────────────────────────────────
def evaluate(case: Case, profile: str) -> None:
    """Fill case.passed and case.fail_reason."""
    active = PROFILE_ACTIVE_INTENTS[profile]
    expected_intent = case.expected_intent

    # Under v1, intents not in the active set route to conversation by design.
    if case.skip_in_v1 and profile == "v1":
        expected_intent = "conversation"
        # In v1 mode, skipped cases pass if they routed to conversation.
        if case.result_intent == "conversation":
            case.passed = True
            return
        case.passed = False
        case.fail_reason = (
            f"intent mismatch: got '{case.result_intent}', expected 'conversation' "
            f"(skipped in v1)"
        )
        return

    # Verify intent classification first
    if case.result_intent != expected_intent:
        case.passed = False
        case.fail_reason = (
            f"intent: got '{case.result_intent}', expected '{expected_intent}'"
        )
        return

    # Verify tool selection
    tools = case.result_tools

    if case.allow_no_tools and not tools:
        case.passed = True
        return

    missing = [t for t in case.required_tools if t not in tools]
    if missing:
        case.passed = False
        case.fail_reason = f"missing tools: {missing} (got: {tools or 'none'})"
        return

    if case.ordered:
        for i in range(len(case.required_tools) - 1):
            a, b = case.required_tools[i], case.required_tools[i + 1]
            try:
                if tools.index(a) > tools.index(b):
                    case.passed = False
                    case.fail_reason = f"order: '{a}' must precede '{b}'"
                    return
            except ValueError:
                pass  # caught by missing check above

    case.passed = True


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_result(n: int, case: Case) -> None:
    num = f"{n:>2}."
    if case.error:
        badge = RED("ERR ")
        detail = case.error[:70]
    elif case.passed:
        badge = GREEN("PASS")
        if case.result_tools:
            detail = f"{case.result_intent} → {', '.join(case.result_tools)}"
        else:
            detail = f"{case.result_intent} (no tools)"
    else:
        badge = RED("FAIL")
        detail = case.fail_reason
    print(f"  {num} [{badge}] {case.query[:55]:<55}  {GREY(detail)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("AI_URL", "http://localhost:5000"),
        help="Base URL for lucid-ai (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--fleet-url",
        default=os.environ.get("FLEET_URL", ""),
        help="Base URL for orchestrator API (default: same as --url)",
    )
    parser.add_argument(
        "--profile",
        choices=["v1", "full"],
        default=os.environ.get("LUCID_AI_TOOLS_PROFILE", "v1"),
        help="Specialist profile to score against (default: v1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PASS_THRESHOLD,
        help=f"Pass threshold (default: {PASS_THRESHOLD})",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    fleet_url = (args.fleet_url or args.url).rstrip("/")
    profile = args.profile
    threshold = args.threshold

    print(BOLD(f"\nLUCID AI accuracy test"))
    print(f"  AI URL:    {base_url}")
    print(f"  Fleet URL: {fleet_url}")
    print(f"  Profile:   {profile}")
    print(f"  Threshold: {threshold:.0%}")
    print("─" * 78)

    with httpx.Client() as client:
        # 1. Probe fleet for real agent IDs
        print("Probing fleet …")
        agents = probe_fleet(client, fleet_url)
        if agents:
            ids = [a["agent_id"] for a in agents]
            print(f"  Found {len(agents)} agent(s): {ids}")
        else:
            print(YELLOW("  No agents found — placeholder IDs will be used"))
        print()

        # 2. Build cases
        cases = build_cases(agents)
        print(f"Running {len(cases)} queries (profile={profile}) …\n")

        # 3. Run each case
        for n, case in enumerate(cases, 1):
            print(f"  {n:>2}. {CYAN(case.query[:70])}")
            try:
                intent, tools, response = run_query(client, base_url, case.query)
                case.result_intent = intent
                case.result_tools = tools
                case.result_response = response
                evaluate(case, profile)
            except Exception as exc:
                case.error = str(exc)
                case.passed = False
            print_result(n, case)

    # 4. Summary
    total    = len(cases)
    errors   = sum(1 for c in cases if c.error)
    passed   = sum(1 for c in cases if c.passed)
    accuracy = passed / total if total else 0.0

    print("\n" + "─" * 78)
    print(BOLD("Results"))
    print(f"  Total queries : {total}")
    print(f"  Errors        : {errors}")
    print(f"  Passed        : {passed}")
    print(f"  Failed        : {total - passed}")
    accuracy_str = f"{accuracy:.0%}"
    if accuracy >= threshold:
        print(f"  Accuracy      : {GREEN(accuracy_str)}  ✓ (threshold: {threshold:.0%})")
    else:
        print(f"  Accuracy      : {RED(accuracy_str)}  ✗ (threshold: {threshold:.0%})")

    # 5. Failure details
    failures = [c for c in cases if not c.passed]
    if failures:
        print(f"\n{BOLD('Failed queries:')}")
        for c in failures:
            label = RED(c.error or c.fail_reason)
            print(f"  • {c.query[:60]}")
            print(f"    {label}")
            print(f"    intent:  {c.result_intent} (expected: {c.expected_intent})")
            print(f"    tools:   {c.result_tools}")
            if c.result_response:
                print(f"    response: {c.result_response[:80]}")

    print()
    return 0 if accuracy >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
