"""Hot-reloadable prompt overrides.

Prompts shipped in ``app/ai/prompts.py`` are the seed defaults. At runtime,
optional overrides can be written to a JSON file (default ``/data/prompts.json``)
and are re-read on every chat request. This lets researchers iterate on prompts
through the Eval UI without rebuilding the container.

Schema of the JSON file::

    {
        "FLEET_SYSTEM_PROMPT": "...override text...",
        "CONVERSATION_SYSTEM_PROMPT": "...",
        ...
    }

If a key is missing or the file is unreadable, the seed default from
``prompts.py`` is used.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

PROMPTS_FILE = Path(os.environ.get("LUCID_AI_PROMPTS_FILE", "/data/prompts.json"))


def load_overrides() -> dict[str, str]:
    """Read prompt overrides from disk. Returns empty dict if file missing/invalid."""
    try:
        text = PROMPTS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        log.warning("Failed to read prompts file %s: %s", PROMPTS_FILE, e)
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Invalid JSON in %s: %s", PROMPTS_FILE, e)
        return {}

    if not isinstance(data, dict):
        log.warning("Prompts file %s is not a dict; ignoring", PROMPTS_FILE)
        return {}

    return {k: v for k, v in data.items() if isinstance(v, str)}


def save_overrides(prompts: dict[str, str]) -> None:
    """Persist prompt overrides to disk."""
    PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve(name: str, default: str) -> str:
    """Return the override for ``name`` if present, else ``default``."""
    return load_overrides().get(name, default)
