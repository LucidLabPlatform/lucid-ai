# lucid-ai — Implementation

> **Package:** `lucid-ai` | **Container:** `lucid-ai` | **Internal port:** 5000

## Overview

`lucid-ai` provides the AI chat interface for LUCID Central Command. It runs a LangGraph ReAct agent powered by a local Ollama LLM (default: `llama3.1:8b`) that can observe fleet state, send commands to agents/components, manage experiments, and delegate tasks to online AI specialist components. Conversation history is persisted to PostgreSQL.

## Key Modules and Responsibilities

| Module | Responsibility |
|--------|----------------|
| `main.py` | FastAPI app creation; lifespan initializes DB schema, `FleetClient`, and `AIWorkflowAgent` |
| `fleet_client.py` | Async HTTP client for `lucid-orchestrator` API; wraps all orchestrator endpoints (agents, commands, experiments, topic links, sync) with error handling that returns strings instead of raising (lets the LLM reason about errors) |
| `db.py` | Postgres schema for AI tables: `conversations`, `conversation_turns`, `researchers`, `researcher_memory`, `lab_memory`; plus full copy of agent/component tables for specialist discovery |
| `ai/supervisor.py` | `AIWorkflowAgent`: LangGraph ReAct agent; builds 22 core tools + dynamic specialist tools; manages conversation context; strips Qwen3 `<think>` blocks from responses |
| `ai/prompts.py` | System prompt template with XML sections; tool triage guide; few-shot examples; anti-loop rules |
| `routes/ai.py` | REST endpoints: `POST /api/ai/chat`, `GET /api/ai/sessions`, `DELETE /api/ai/sessions/{id}`, `GET /api/ai/history` |

## Important Implementation Details

### ReAct Agent Architecture

The agent uses the LangGraph `create_react_agent` pattern:

1. **Per-call agent construction** — A new ReAct agent with a fresh `MemorySaver` checkpoint is created for each chat call. This ensures the conversation history injected via the message list matches the Postgres-stored history.
2. **System prompt injection** — The prompt includes a live fleet summary (agent IDs, statuses, component IDs) and a list of online specialists, both fetched at call time.
3. **Tool categories**:
   - **Core tools** (22): `list_agents`, `get_agent`, `send_agent_command`, `send_component_command`, `get_command_catalog`, experiment management (7 tools), topic link management (6 tools), logs/commands, sync state, `delete_agent`
   - **Specialist tools** (dynamic): One tool per online component with `capabilities.type == "ai_specialist"`. Each publishes `cmd/task` via the orchestrator's internal command endpoint.
4. **Error resilience** — Every core tool is wrapped in a `safe_coro` decorator that catches exceptions and returns error strings instead of crashing the ReAct loop.

### Specialist Tool Discovery

At each chat call, `DB.get_available_specialists()` queries `component_metadata` for components whose `capabilities` JSON contains `type: "ai_specialist"` and whose parent agent has status `running`. For each specialist:

- A `@tool`-decorated async function is created with closure-captured `agent_id` and `component_id` (using Python default argument capture to avoid the loop variable pitfall).
- The tool publishes `cmd/task` with `{"task": "<user prompt>"}` via the fleet client's `send_command(wait=True)`.
- Timeout (30s) and errors return user-friendly strings.

### Prompt Engineering

The system prompt (`SUPERVISOR_SYSTEM_PROMPT`) is designed for Qwen3-class models with `/no_think` mode:

- **XML sections** for clear boundary recognition
- **Tool triage block** mapping intent categories to tool groups
- **Few-shot examples** covering the five most common query patterns
- **Anti-loop rules** preventing redundant tool calls
- **Fleet and specialist injection** via `{fleet_summary}` and `{specialist_list}` placeholders

### Conversation Persistence

- `DB.upsert_conversation(session_id)` ensures a conversation record exists
- `DB.get_conversation_turns(session_id)` returns `[(role, content)]` tuples
- `DB.save_conversation_turns(session_id, user_msg, assistant_msg)` persists both turns
- Sessions can be listed (`GET /api/ai/sessions`) and deleted (`DELETE /api/ai/sessions/{id}`)

### Payload Coercion

`_coerce_payload()` handles the fact that LLMs may pass payloads as dicts, JSON strings, empty strings, None, or other types. It normalizes all cases to a dict for downstream MQTT calls.

## How It Connects to Other Services

- **lucid-orchestrator** — All fleet operations (commands, experiments, topic links) go through the `FleetClient` HTTP client targeting the orchestrator's internal API
- **Ollama** — LLM inference via `ChatOllama` (LangChain wrapper)
- **PostgreSQL** — Direct psycopg2 connections for conversation history and specialist discovery
- **lucid-ui** — Receives proxied requests from the UI at `/api/ai/*`
