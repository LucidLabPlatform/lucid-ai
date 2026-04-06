# lucid-ai

AI supervisor service for LUCID.

Owns:
- `/api/ai/chat`
- `/api/ai/history`
- `/health`
- conversation persistence and specialist discovery

Uses `lucid-orchestrator` for:
- fleet inspection
- experiment template/run control
- MQTT-facing task dispatch to specialist components

Runtime requirements:
- Postgres via `LUCID_DB_URL`
- Orchestrator via `LUCID_FLEET_CORE_URL`
- Ollama via `OLLAMA_BASE_URL`
