"""Flat-file storage (R6 — no DB). Three concerns:

- `data/fetched/{url_hash}.json` — raw fetched + metadata; doubles as a cache
- `data/briefings/{timestamp}-{slug}.md` — final agent outputs
- `data/logs/runs.jsonl` — one line per run; greppable observability (D6)
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FETCHED = DATA / "fetched"
BRIEFINGS = DATA / "briefings"
LOGS = DATA / "logs"

for _p in (FETCHED, BRIEFINGS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)


def url_hash(url: str) -> str:
    """SHA-256 of URL, truncated to 16 hex chars. Deterministic."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def fetched_path(url: str) -> Path:
    return FETCHED / f"{url_hash(url)}.json"


def save_fetched(url: str, payload: dict) -> Path:
    path = fetched_path(url)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def load_fetched(url: str) -> dict | None:
    """Return cached payload for URL, or None if never fetched."""
    path = fetched_path(url)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def slugify(text: str, max_len: int = 50) -> str:
    """Lowercase, kebab-cased, length-capped. Never empty."""
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:max_len] or "briefing"


def save_briefing(prompt: str, markdown: str) -> Path:
    """Persist briefing to data/briefings/{ts}-{slug}.md. Returns the Path.

    Timestamp includes microseconds (%f) so two runs of the same prompt within
    one second don't silently overwrite each other.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = BRIEFINGS / f"{ts}-{slugify(prompt)}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def log_run(entry: dict) -> None:
    """Append one JSON line to data/logs/runs.jsonl. Adds ISO timestamp."""
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with (LOGS / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_event(level: str, message: str, **fields) -> None:
    """Append one human-readable log line to data/logs/agent.log.

    Companion to runs.jsonl — runs.jsonl is the machine-readable run record;
    agent.log is the streaming narration ("what is the agent doing right now").
    In production this would graduate to structured logging (Langfuse / OTel);
    here it's a single function call to demonstrate the pattern without the
    vendor footprint.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{ts}] {level:>5} {message}" + (f"  {extras}" if extras else "")
    with (LOGS / "agent.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
