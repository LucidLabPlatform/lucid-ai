#!/usr/bin/env python3
"""LUCID AI accuracy test — Day 11 deliverable.

Sends 20 representative queries to POST /api/ai/chat and checks that the
correct tools were called (and in the correct order where it matters).

The target is the lucid-ui proxy on port 5000, which forwards /api/ai/* to
the lucid-ai service. Run this while the full stack is up:

    cd lucid-central-command
    docker compose up -d
    python lucid-ai/tests/ai_accuracy_test.py

Options:
    --url   Base URL (default: http://localhost:5000 via lucid-ui proxy,
            or AI_URL env var)

Exit code:
    0  accuracy >= 80%
    1  accuracy <  80%
"""

import argparse
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
BOLD   = lambda t: _c("\033[1m",  t)  # noqa: E731

PASS_THRESHOLD = 0.80


# ── Test case definition ──────────────────────────────────────────────────────
@dataclass
class Case:
    """One test query with expected tool-call criteria."""
    query: str
    required_tools: list[str]
    # If True, required_tools must appear in that exact order.
    ordered: bool = False
    description: str = ""
    # If True, the case passes even when no tools are called — the model is
    # allowed to answer directly from the fleet_summary in the system prompt.
    allow_no_tools: bool = False
    # Filled in at runtime.
    result_tools: list[str] = field(default_factory=list)
    passed: bool | None = None
    fail_reason: str = ""
    error: str = ""


# ── Fleet probe ───────────────────────────────────────────────────────────────
def probe_fleet(client: httpx.Client, base_url: str) -> list[dict]:
    """Return agent list from lucid-orchestrator via lucid-ui proxy."""
    try:
        resp = client.get(f"{base_url}/api/agents", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # API may return a list or {"agents": [...]}
        return data if isinstance(data, list) else data.get("agents", [])
    except Exception as exc:
        print(YELLOW(f"  Warning: could not probe fleet ({exc}) — using placeholder IDs"))
        return []


# ── Case builder ──────────────────────────────────────────────────────────────
def build_cases(agents: list[dict]) -> list[Case]:
    """Build the 20 test cases, substituting real agent IDs where available."""

    # Resolve real agent/component IDs from the live fleet.
    agent_ids = [a["agent_id"] for a in agents]

    def pick_agent(keyword: str, fallback: str) -> str:
        for a in agents:
            comps = list((a.get("components") or {}).keys())
            if any(keyword in c for c in comps):
                return a["agent_id"]
        return agent_ids[0] if agent_ids else fallback

    first_agent  = agent_ids[0] if agent_ids else "robot-01"
    led_agent    = pick_agent("led",     "led-agent-01")
    robot_agent  = pick_agent("ros",     "robot-agent-01")
    camera_agent = pick_agent("camera",  first_agent)

    has_agents = bool(agent_ids)

    return [
        # ── Fleet queries (4) ────────────────────────────────────────────────
        Case(
            query="What devices are online?",
            required_tools=["list_agents"],
            allow_no_tools=True,
            description="fleet status (answerable from fleet context)",
        ),
        Case(
            query=f"Show me the full state of {first_agent}",
            required_tools=["get_agent"],
            description="single agent detail",
        ),
        Case(
            query=f"What components does {first_agent} have?",
            required_tools=["list_agents"],
            allow_no_tools=True,
            description="component enumeration (answerable from fleet context)",
        ),
        Case(
            query="Are any agents currently offline?",
            required_tools=["list_agents"],
            allow_no_tools=True,
            description="offline check (answerable from fleet context)",
        ),

        # ── Hardware commands (4) — catalog must precede send ────────────────
        Case(
            query=f"Set the LED strip on {led_agent} to red",
            required_tools=["get_command_catalog", "send_component_command"],
            ordered=True,
            description="command: catalog → send (colour)",
        ),
        Case(
            query=f"Turn off the LED strip on {led_agent}",
            required_tools=["get_command_catalog", "send_component_command"],
            ordered=True,
            description="command: catalog → send (off)",
        ),
        Case(
            query=f"Set LED strip to blue on {led_agent}",
            required_tools=["get_command_catalog", "send_component_command"],
            ordered=True,
            description="command: catalog → send (colour, rephrased)",
        ),
        Case(
            query=f"What commands can I send to {first_agent}?",
            required_tools=["get_command_catalog"],
            description="command discovery only",
        ),

        # ── Experiments (4) ──────────────────────────────────────────────────
        Case(
            query="What experiment templates are available?",
            required_tools=["list_experiment_templates"],
            description="template discovery",
        ),
        Case(
            query="Run the foraging experiment",
            required_tools=["list_experiment_templates", "configure_experiment", "start_experiment", "get_experiment_run"],
            ordered=True,
            description="discover → configure → start → confirm status",
        ),
        Case(
            query="Show me the most recent experiment run",
            required_tools=["list_experiment_runs"],
            description="run history",
        ),
        Case(
            query="What's the status of the last experiment?",
            required_tools=["list_experiment_runs"],
            description="run status (rephrased)",
        ),

        # ── Topic links (4) ──────────────────────────────────────────────────
        Case(
            query="Show me all topic links",
            required_tools=["list_topic_links"],
            description="list routing rules",
        ),
        Case(
            query=(
                f"Link robot perception on {robot_agent} to the LED strip on {led_agent}"
            ),
            required_tools=["create_topic_link"],
            description="create EMQX republish rule",
        ),
        Case(
            query="Deactivate all active topic links",
            required_tools=["list_topic_links", "deactivate_topic_link"],
            ordered=True,
            description="discover IDs → deactivate each",
        ),
        Case(
            query="How many topic links are currently active?",
            required_tools=["list_topic_links"],
            description="link count query",
        ),

        # ── Metrics / logs (4) ───────────────────────────────────────────────
        Case(
            query="What's the throughput on the perception link?",
            required_tools=["list_topic_links", "get_topic_link"],
            ordered=True,
            description="list → get metrics",
        ),
        Case(
            query=f"Show me recent logs from {first_agent}",
            required_tools=["get_agent_logs"],
            description="agent log retrieval",
        ),
        Case(
            query=f"What commands were sent to {first_agent} recently?",
            required_tools=["get_agent_commands"],
            description="command history",
        ),
        Case(
            query=f"Show me the command history for {camera_agent}",
            required_tools=["get_agent_commands"],
            description="command history (rephrased)",
        ),
    ]


# ── Query runner ──────────────────────────────────────────────────────────────
def run_query(client: httpx.Client, base_url: str, query: str) -> tuple[list[str], str]:
    """POST to /api/ai/chat; return (tool_names_in_order, response_text)."""
    resp = client.post(
        f"{base_url}/api/ai/chat",
        json={"message": query, "session_id": str(uuid.uuid4())},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    tool_names = [tc.get("name", "") for tc in data.get("tool_calls", [])]
    return tool_names, data.get("response", "")


# ── Pass/fail logic ───────────────────────────────────────────────────────────
def evaluate(case: Case) -> None:
    """Fill case.passed and case.fail_reason from case.result_tools."""
    tools = case.result_tools
    # Case is allowed to answer directly from system-prompt context.
    if case.allow_no_tools and not tools:
        case.passed = True
        return
    missing = [t for t in case.required_tools if t not in tools]
    if missing:
        case.passed = False
        case.fail_reason = f"missing: {missing}"
        return
    if case.ordered:
        for i in range(len(case.required_tools) - 1):
            a, b = case.required_tools[i], case.required_tools[i + 1]
            try:
                if tools.index(a) > tools.index(b):
                    case.passed = False
                    case.fail_reason = f"order: {a} must precede {b}"
                    return
            except ValueError:
                pass  # already caught by missing check above
    case.passed = True


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_result(n: int, case: Case) -> None:
    num = f"{n:>2}."
    if case.error:
        badge = RED("ERR ")
        detail = case.error[:60]
    elif case.passed:
        badge = GREEN("PASS")
        detail = ", ".join(case.result_tools) or "(answered from context)"
    else:
        badge = RED("FAIL")
        detail = case.fail_reason
    print(f"  {num} [{badge}] {case.query[:55]:<55}  {detail}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("AI_URL", "http://localhost:5000"),
        help="Base URL for the LUCID stack (default: http://localhost:5000)",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(BOLD(f"\nLUCID AI accuracy test — target: {base_url}"))
    print("─" * 75)

    with httpx.Client() as client:
        # 1. Probe fleet for real agent IDs
        print("Probing fleet …")
        agents = probe_fleet(client, base_url)
        if agents:
            ids = [a["agent_id"] for a in agents]
            print(f"  Found {len(agents)} agent(s): {ids}")
        else:
            print(YELLOW("  No agents found — placeholder IDs will be used in queries"))
        print()

        # 2. Build cases with real IDs where possible
        cases = build_cases(agents)
        assert len(cases) == 20, f"Expected 20 cases, got {len(cases)}"

        # 3. Run each case
        print(f"Running {len(cases)} queries …\n")
        for n, case in enumerate(cases, 1):
            print(f"  {n:>2}. {CYAN(case.query[:68])}")
            try:
                tools, _response = run_query(client, base_url, case.query)
                case.result_tools = tools
                evaluate(case)
            except Exception as exc:
                case.error = str(exc)
                case.passed = False
            print_result(n, case)

    # 4. Summary
    total    = len(cases)
    errors   = sum(1 for c in cases if c.error)
    passed   = sum(1 for c in cases if c.passed)
    accuracy = passed / total if total else 0.0

    print("\n" + "─" * 75)
    print(BOLD("Results"))
    print(f"  Total queries : {total}")
    print(f"  Errors        : {errors}")
    print(f"  Passed        : {passed}")
    print(f"  Failed        : {total - passed}")
    accuracy_str = f"{accuracy:.0%}"
    if accuracy >= PASS_THRESHOLD:
        print(f"  Accuracy      : {GREEN(accuracy_str)}  ✓ (threshold: {PASS_THRESHOLD:.0%})")
    else:
        print(f"  Accuracy      : {RED(accuracy_str)}  ✗ (threshold: {PASS_THRESHOLD:.0%})")
        print(RED("\n  Accuracy below threshold — consider refining the system prompt or"))
        print(RED("  switching to a larger model (e.g. qwen3:32b)."))

    # 5. Failure details
    failures = [c for c in cases if not c.passed]
    if failures:
        print(f"\n{BOLD('Failed queries:')}")
        for c in failures:
            label = RED(c.error or c.fail_reason)
            print(f"  • {c.query[:60]}")
            print(f"    {label}")
            print(f"    got: {c.result_tools}")
            print(f"    expected: {c.required_tools}")

    print()
    return 0 if accuracy >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
