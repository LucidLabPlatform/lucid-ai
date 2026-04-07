"""Shared Postgres connection and schema initialisation for lucid-cc.

Public API
----------
connect([url])
    Open a psycopg2 connection, retrying up to 10 times for Postgres start-up.
init_schema([url])
    Create all tables and indexes (idempotent; uses ``CREATE … IF NOT EXISTS``).
upsert_agent(conn, agent_id, ts)
    Insert or update the ``agents`` registry row.
upsert_component(conn, agent_id, component_id, ts)
    Insert or update the ``components`` registry row.
get_available_specialists()
    Query component_metadata for online AI specialist components.
get_conversation_turns(session_id)
    Return conversation history as ``[(role, content)]`` tuples.
upsert_conversation(session_id)
    Ensure a conversation record exists, return its id.
save_conversation_turns(session_id, user_msg, assistant_msg)
    Persist a user + assistant turn pair to the DB.

Schema groups
-------------
* **Auth** — ``users``, ``authn_log``, ``authz_log``
* **Agents** — ``agents``, ``agent_status/state/metadata/cfg*``,
  ``agent_logs/telemetry/events``, ``commands``
* **Components** — ``components``, ``component_status/state/metadata/cfg*``,
  ``component_logs/telemetry/events``
* **AI / Research** — ``researchers``, ``conversations``,
  ``conversation_turns``, ``researcher_memory``, ``lab_memory``
* **Experiments** — ``experiment_templates``, ``experiment_runs``,
  ``experiment_steps``, ``experiment_topic_links``

The module reads ``LUCID_DB_URL`` from the environment (default:
``postgresql://lucid:lucid_secret@localhost:5432/lucid``).
"""
import json
import os
import time
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras

DB_URL = os.environ["LUCID_DB_URL"]


def connect(url: str | None = None) -> psycopg2.extensions.connection:
    """Open a new psycopg2 connection, retrying up to 10× for Postgres start-up."""
    target = url or DB_URL
    for attempt in range(10):
        try:
            return psycopg2.connect(target)
        except psycopg2.OperationalError:
            if attempt == 9:
                raise
            time.sleep(1)
    raise RuntimeError("unreachable")


def init_schema(url: str | None = None) -> None:
    """Create all tables and indexes (idempotent). Called at service start-up."""
    with connect(url) as conn:
        with conn.cursor() as cur:

            # ----------------------------------------------------------------
            # Auth
            # ----------------------------------------------------------------

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username      TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS authn_log (
                    id       BIGSERIAL PRIMARY KEY,
                    ts       TIMESTAMPTZ NOT NULL,
                    username TEXT NOT NULL,
                    clientid TEXT NOT NULL,
                    result   TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS authz_log (
                    id       BIGSERIAL PRIMARY KEY,
                    ts       TIMESTAMPTZ NOT NULL,
                    username TEXT NOT NULL,
                    clientid TEXT NOT NULL,
                    topic    TEXT NOT NULL,
                    action   TEXT NOT NULL,
                    result   TEXT NOT NULL
                )
            """)

            # ----------------------------------------------------------------
            # Agents — core registry
            # ----------------------------------------------------------------

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id      TEXT PRIMARY KEY,
                    first_seen_ts TIMESTAMPTZ NOT NULL,
                    last_seen_ts  TIMESTAMPTZ NOT NULL
                )
            """)

            # ── agent retained topics (one row per agent, upserted)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_status (
                    agent_id           TEXT PRIMARY KEY REFERENCES agents,
                    state              TEXT,
                    connected_since_ts TIMESTAMPTZ,
                    uptime_s           FLOAT,
                    version            TEXT,
                    received_ts        TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_state (
                    agent_id       TEXT PRIMARY KEY REFERENCES agents,
                    cpu_percent    FLOAT,
                    memory_percent FLOAT,
                    disk_percent   FLOAT,
                    components     JSONB,
                    received_ts    TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_metadata (
                    agent_id     TEXT PRIMARY KEY REFERENCES agents,
                    version      TEXT,
                    platform     TEXT,
                    architecture TEXT,
                    received_ts  TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_cfg (
                    agent_id    TEXT PRIMARY KEY REFERENCES agents,
                    heartbeat_s INTEGER,
                    received_ts TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_cfg_logging (
                    agent_id    TEXT PRIMARY KEY REFERENCES agents,
                    log_level   TEXT,
                    received_ts TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_cfg_telemetry (
                    agent_id              TEXT PRIMARY KEY REFERENCES agents,
                    cpu_pct_enabled       BOOLEAN,
                    cpu_pct_interval_s    INTEGER,
                    cpu_pct_threshold     FLOAT,
                    memory_pct_enabled    BOOLEAN,
                    memory_pct_interval_s INTEGER,
                    memory_pct_threshold  FLOAT,
                    disk_pct_enabled      BOOLEAN,
                    disk_pct_interval_s   INTEGER,
                    disk_pct_threshold    FLOAT,
                    received_ts           TIMESTAMPTZ NOT NULL
                )
            """)

            # ── agent streaming / events

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_logs (
                    id          BIGSERIAL PRIMARY KEY,
                    agent_id    TEXT NOT NULL,
                    ts          TIMESTAMPTZ NOT NULL,
                    level       TEXT NOT NULL,
                    logger      TEXT,
                    message     TEXT NOT NULL,
                    exception   TEXT,
                    received_ts TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_telemetry (
                    id          BIGSERIAL PRIMARY KEY,
                    agent_id    TEXT NOT NULL,
                    metric      TEXT NOT NULL,
                    value       FLOAT NOT NULL,
                    received_ts TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_events (
                    id          BIGSERIAL PRIMARY KEY,
                    agent_id    TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    request_id  TEXT NOT NULL,
                    ok          BOOLEAN NOT NULL,
                    error       TEXT,
                    received_ts TIMESTAMPTZ NOT NULL
                )
            """)

            # ── commands

            cur.execute("""
                CREATE TABLE IF NOT EXISTS commands (
                    id             BIGSERIAL PRIMARY KEY,
                    sent_ts        TIMESTAMPTZ NOT NULL,
                    request_id     TEXT NOT NULL UNIQUE,
                    agent_id       TEXT NOT NULL,
                    component_id   TEXT,
                    action         TEXT NOT NULL,
                    topic          TEXT NOT NULL,
                    payload        JSONB,
                    result_ok      BOOLEAN,
                    result_ts      TIMESTAMPTZ,
                    result_payload JSONB
                )
            """)

            # ----------------------------------------------------------------
            # Components — nested under agents
            # ----------------------------------------------------------------

            cur.execute("""
                CREATE TABLE IF NOT EXISTS components (
                    agent_id      TEXT NOT NULL,
                    component_id  TEXT NOT NULL,
                    first_seen_ts TIMESTAMPTZ NOT NULL,
                    last_seen_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id)
                )
            """)

            # ── component retained topics

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_status (
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    state        TEXT,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id),
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_metadata (
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    version      TEXT,
                    capabilities JSONB,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id),
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_state (
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    payload      JSONB,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id),
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_cfg (
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    payload      JSONB,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id),
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_cfg_logging (
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    log_level    TEXT,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id),
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_cfg_telemetry (
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    payload      JSONB,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (agent_id, component_id),
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            # ── component streaming / events

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_logs (
                    id           BIGSERIAL PRIMARY KEY,
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    level        TEXT NOT NULL,
                    message      TEXT NOT NULL,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_telemetry (
                    id           BIGSERIAL PRIMARY KEY,
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    metric       TEXT NOT NULL,
                    value        JSONB NOT NULL,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS component_events (
                    id           BIGSERIAL PRIMARY KEY,
                    agent_id     TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    action       TEXT NOT NULL,
                    request_id   TEXT NOT NULL,
                    ok           BOOLEAN NOT NULL,
                    applied      JSONB,
                    error        TEXT,
                    received_ts  TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (agent_id, component_id) REFERENCES components
                )
            """)

            # ----------------------------------------------------------------
            # AI / research tables (used from Day 8 onward)
            # ----------------------------------------------------------------

            cur.execute("""
                CREATE TABLE IF NOT EXISTS researchers (
                    id            TEXT PRIMARY KEY,
                    display_name  TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    voice_profile BYTEA,
                    preferences   JSONB,
                    created_at    TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id             TEXT PRIMARY KEY,
                    researcher_id  TEXT NOT NULL REFERENCES researchers,
                    started_at     TIMESTAMPTZ NOT NULL,
                    last_active_at TIMESTAMPTZ NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id              BIGSERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations,
                    ts              TIMESTAMPTZ NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    interface       TEXT,
                    reasoning       TEXT,
                    actions_taken   JSONB
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS researcher_memory (
                    id            BIGSERIAL PRIMARY KEY,
                    researcher_id TEXT NOT NULL REFERENCES researchers,
                    key           TEXT NOT NULL,
                    value         TEXT NOT NULL,
                    updated_at    TIMESTAMPTZ NOT NULL,
                    UNIQUE (researcher_id, key)
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS lab_memory (
                    id         BIGSERIAL PRIMARY KEY,
                    key        TEXT NOT NULL UNIQUE,
                    value      TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
            """)

            # ----------------------------------------------------------------
            # Experiment engine
            # ----------------------------------------------------------------

            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_templates (
                    id                TEXT PRIMARY KEY,
                    name              TEXT NOT NULL,
                    version           TEXT NOT NULL DEFAULT '1.0.0',
                    description       TEXT NOT NULL DEFAULT '',
                    parameters_schema JSONB NOT NULL DEFAULT '{}',
                    definition        JSONB NOT NULL,
                    tags              TEXT[] NOT NULL DEFAULT '{}',
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id          TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL REFERENCES experiment_templates,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    parameters  JSONB NOT NULL DEFAULT '{}',
                    started_at  TIMESTAMPTZ,
                    ended_at    TIMESTAMPTZ,
                    error       TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_steps (
                    id               BIGSERIAL PRIMARY KEY,
                    run_id           TEXT NOT NULL REFERENCES experiment_runs,
                    step_index       INTEGER NOT NULL,
                    step_name        TEXT NOT NULL,
                    agent_id         TEXT,
                    component_id     TEXT,
                    action           TEXT,
                    request_payload  JSONB,
                    response_payload JSONB,
                    status           TEXT NOT NULL DEFAULT 'pending',
                    attempt          INTEGER NOT NULL DEFAULT 0,
                    started_at       TIMESTAMPTZ,
                    ended_at         TIMESTAMPTZ,
                    duration_ms      INTEGER
                )
            """)

            # ----------------------------------------------------------------
            # Topic links — EMQX Rule Engine backed bridges
            # ----------------------------------------------------------------

            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_topic_links (
                    id                TEXT PRIMARY KEY,
                    name              TEXT NOT NULL,
                    source_topic      TEXT NOT NULL,
                    target_topic      TEXT NOT NULL,
                    select_clause     TEXT NOT NULL DEFAULT '*',
                    payload_template  TEXT,
                    qos               INTEGER NOT NULL DEFAULT 0,
                    emqx_rule_id      TEXT,
                    enabled           BOOLEAN NOT NULL DEFAULT true,
                    experiment_run_id TEXT REFERENCES experiment_runs ON DELETE SET NULL,
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

            # ----------------------------------------------------------------
            # Indexes
            # ----------------------------------------------------------------

            # auth
            cur.execute("CREATE INDEX IF NOT EXISTS idx_authn_log_ts         ON authn_log (ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_authn_log_username    ON authn_log (username)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_authz_log_ts         ON authz_log (ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_authz_log_username    ON authz_log (username)")

            # agent streaming
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_ts        ON agent_logs (agent_id, received_ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_telemetry_agent_ts   ON agent_telemetry (agent_id, received_ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_telemetry_metric_ts  ON agent_telemetry (agent_id, metric, received_ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_events_request_id    ON agent_events (request_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_commands_agent_ts          ON commands (agent_id, sent_ts DESC)")

            # component streaming
            cur.execute("CREATE INDEX IF NOT EXISTS idx_comp_logs_agent_comp_ts      ON component_logs (agent_id, component_id, received_ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_comp_telemetry_agent_comp_ts ON component_telemetry (agent_id, component_id, received_ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_comp_events_agent_comp_ts    ON component_events (agent_id, component_id, received_ts DESC)")

            # AI / research
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_turns_conv_ts        ON conversation_turns (conversation_id, ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_researcher_memory_rid_key ON researcher_memory (researcher_id, key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_researcher  ON conversations (researcher_id)")

            # experiment engine
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exp_runs_template   ON experiment_runs (template_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exp_runs_status     ON experiment_runs (status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exp_steps_run_idx   ON experiment_steps (run_id, step_index)")

            # topic links
            cur.execute("CREATE INDEX IF NOT EXISTS idx_topic_links_run     ON experiment_topic_links (experiment_run_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_topic_links_enabled ON experiment_topic_links (enabled)")

        conn.commit()


def upsert_agent(conn: psycopg2.extensions.connection, agent_id: str, ts: datetime) -> None:
    """Insert a new agent row or update ``last_seen_ts`` if it already exists.

    Args:
        conn:     Open psycopg2 connection (caller is responsible for commit).
        agent_id: Unique agent identifier (e.g. ``"robot-01"``).
        ts:       Timestamp for ``first_seen_ts`` (on insert) and ``last_seen_ts``.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agents (agent_id, first_seen_ts, last_seen_ts)
            VALUES (%s, %s, %s)
            ON CONFLICT (agent_id) DO UPDATE SET last_seen_ts = EXCLUDED.last_seen_ts
            """,
            (agent_id, ts, ts),
        )


def upsert_component(conn: psycopg2.extensions.connection, agent_id: str,
                     component_id: str, ts: datetime) -> None:
    """Insert a new component row or update ``last_seen_ts`` if it already exists.

    The ``components`` table has a composite primary key ``(agent_id, component_id)``
    with a FK to ``agents``, so ``upsert_agent`` must be called first within
    the same transaction.

    Args:
        conn:         Open psycopg2 connection (caller is responsible for commit).
        agent_id:     Agent that owns this component.
        component_id: Unique component identifier within the agent.
        ts:           Timestamp for ``first_seen_ts`` (on insert) and ``last_seen_ts``.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO components (agent_id, component_id, first_seen_ts, last_seen_ts)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (agent_id, component_id) DO UPDATE SET last_seen_ts = EXCLUDED.last_seen_ts
            """,
            (agent_id, component_id, ts, ts),
        )


def json_dumps(obj: Any) -> str:
    """Serialize ``obj`` to a JSON string.  Thin wrapper kept for symmetry."""
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# AI / conversation helpers
# ---------------------------------------------------------------------------

_AI_RESEARCHER_ID = "__ai_chat__"


def _ensure_ai_researcher(conn: psycopg2.extensions.connection) -> None:
    """Ensure the sentinel researcher row exists for AI chat sessions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO researchers (id, display_name, role, created_at)
            VALUES (%s, 'AI Chat', 'ai', NOW())
            ON CONFLICT (id) DO NOTHING
            """,
            (_AI_RESEARCHER_ID,),
        )


def get_available_specialists() -> list[dict]:
    """Query component_metadata for ai_specialist type components on online agents."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.agent_id, cm.component_id,
                       cm.capabilities->>'description' AS description
                FROM component_metadata cm
                JOIN agents a ON cm.agent_id = a.agent_id
                LEFT JOIN agent_status s ON s.agent_id = a.agent_id
                WHERE cm.capabilities->>'type' = 'ai_specialist'
                  AND s.state = 'online'
                ORDER BY cm.component_id
            """)
            rows = cur.fetchall()
    return [
        {"agent_id": r[0], "component_id": r[1], "description": r[2] or ""}
        for r in rows
    ]


def list_conversations() -> list[dict]:
    """Return all AI chat sessions, most recent first."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.started_at, c.last_active_at,
                       (SELECT ct.content FROM conversation_turns ct
                        WHERE ct.conversation_id = c.id AND ct.role = 'user'
                        ORDER BY ct.ts ASC LIMIT 1) AS first_message
                FROM conversations c
                WHERE c.researcher_id = %s
                ORDER BY c.last_active_at DESC
            """, (_AI_RESEARCHER_ID,))
            return [
                {
                    "session_id": r[0],
                    "started_at": r[1].isoformat() if r[1] else None,
                    "last_active_at": r[2].isoformat() if r[2] else None,
                    "preview": (r[3] or "")[:80],
                }
                for r in cur.fetchall()
            ]


def get_conversation_turns(session_id: str) -> list[tuple[str, str]]:
    """Return conversation turns as list of (role, content) tuples, oldest first."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ct.role, ct.content
                FROM conversation_turns ct
                WHERE ct.conversation_id = %s
                ORDER BY ct.ts ASC
            """, (session_id,))
            return [(r[0], r[1]) for r in cur.fetchall()]


def upsert_conversation(session_id: str) -> str:
    """Ensure a conversation record exists for this session_id, return its id."""
    with connect() as conn:
        _ensure_ai_researcher(conn)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (id, researcher_id, started_at, last_active_at)
                VALUES (%s, %s, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET last_active_at = NOW()
                RETURNING id
            """, (session_id, _AI_RESEARCHER_ID))
            result = cur.fetchone()[0]
        conn.commit()
    return result


def save_conversation_turns(
    session_id: str, user_msg: str, assistant_msg: str
) -> None:
    """Save a user+assistant exchange to the DB."""
    with connect() as conn:
        _ensure_ai_researcher(conn)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (id, researcher_id, started_at, last_active_at)
                VALUES (%s, %s, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET last_active_at = NOW()
            """, (session_id, _AI_RESEARCHER_ID))
            cur.execute("""
                INSERT INTO conversation_turns (conversation_id, role, content, ts)
                VALUES (%s, 'user', %s, NOW()), (%s, 'assistant', %s, NOW())
            """, (session_id, user_msg, session_id, assistant_msg))
        conn.commit()
