"""Self-evaluation after every run. Three layers:

1. Structure check — deterministic regex over the briefing markdown.
2. Trace critique — scan tool_calls for known anti-patterns from runs.jsonl.
3. LLM judge — one Haiku call for usefulness + next-run suggestion.

Output appended to `data/logs/evals.jsonl`. Compounding works via
`demonstrations_block()` (U3): few-shot examples from kept/reverted edits
gated behind metric_version=composite-v1 records. The legacy
`load_recent_lessons()` is preserved for backward compat but no longer the
primary compounding mechanism.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent import llm, storage

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_LESSONS = 5  # how many recent lessons to surface in SYSTEM_PROMPT


# ---------- Layer 1: deterministic structure check ----------

def check_structure(briefing: str) -> dict:
    """Regex checks over the markdown briefing. No LLM, no I/O.

    Returns {passes: [str], fails: [str], score: int_0_to_5}.
    """
    passes, fails = [], []

    if re.search(r"^##\s+TL;DR", briefing, re.MULTILINE):
        passes.append("tldr_section_present")
    else:
        fails.append("tldr_section_missing")

    themes_block = re.search(
        r"^##\s+Key themes\s*\n(.*?)(?=\n##\s|\Z)", briefing, re.MULTILINE | re.DOTALL
    )
    theme_bullets = re.findall(r"^-\s+.+$", themes_block.group(1), re.MULTILINE) if themes_block else []
    if len(theme_bullets) >= 3:
        passes.append(f"themes_count_ok ({len(theme_bullets)})")
    else:
        fails.append(f"themes_count_low ({len(theme_bullets)})")

    citation_bullets = [b for b in theme_bullets if re.search(r"\[\^\d+\]", b)]
    if theme_bullets and len(citation_bullets) == len(theme_bullets):
        passes.append("all_themes_cited")
    elif theme_bullets:
        fails.append(f"uncited_themes ({len(theme_bullets) - len(citation_bullets)})")

    if re.search(r"^##\s+Sentiment\s*\n\s*\*\*(Positive|Neutral|Negative)\*\*", briefing, re.MULTILINE):
        passes.append("sentiment_labeled")
    else:
        fails.append("sentiment_missing_or_unlabeled")

    if re.search(r"^##\s+Sources\s*\n.*?\[\^\d+\]:", briefing, re.MULTILINE | re.DOTALL):
        passes.append("sources_section_present")
    else:
        fails.append("sources_section_missing")

    score = len(passes)  # 0-5
    return {"passes": passes, "fails": fails, "score": score}


# ---------- Layer 2: trace critique ----------

def critique_trace(run_log: dict) -> list[dict]:
    """Scan run_log['tool_calls'] for known anti-patterns. Returns a list of
    structured issues. Each issue: {tag, detail, suggestion}.
    """
    calls = run_log.get("tool_calls", [])
    issues: list[dict] = []

    summarize_errors = [c for c in calls if c["name"] == "summarize" and c.get("error")]
    summarize_ok = [c for c in calls if c["name"] == "summarize" and not c.get("error")]
    if summarize_errors:
        bad_args = [c for c in summarize_errors if "missing" in (c.get("error") or "") and "documents" in (c.get("error") or "")]
        if bad_args:
            # Suggestion updated post-U0 (review AC-3): the new tool contract
            # has `documents` as OPTIONAL — the loop auto-attaches. If the
            # `missing documents` error still fires after U0, it means the
            # auto-attach hook itself failed (not a model-side prompting
            # issue). Don't suggest re-enforcing the old required-documents
            # contract — that would create a self-defeating feedback loop
            # where meta-eval proposes SYSTEM_PROMPT edits to undo U0.
            issues.append({
                "tag": "summarize_missing_documents",
                "detail": f"summarize called {len(bad_args)} time(s) without `documents` arg",
                "suggestion": "Investigate why the loop's auto-attach hook (loop._collect_fetched_docs) did not inject documents. This is a code path issue, not a prompting issue — `documents` is OPTIONAL per the current tool schema.",
            })

    search_count = sum(1 for c in calls if c["name"] == "search")
    if search_count > 4:
        issues.append({
            "tag": "search_budget_exceeded",
            "detail": f"{search_count} search calls (budget = 3-4)",
            "suggestion": "Stop searching once you have 4 distinct relevant URLs. Move to fetch, then summarize.",
        })

    fetch_403 = [c for c in calls if c["name"] == "fetch" and "403" in (c.get("error") or "")]
    if fetch_403:
        issues.append({
            "tag": "fetch_403_dropped",
            "detail": f"{len(fetch_403)} fetch(es) returned 403",
            "suggestion": "When a domain 403s, do not retry — pick a different source from search results.",
        })

    if run_log.get("iterations") == 12 and not summarize_ok:
        issues.append({
            "tag": "cap_hit_without_summarize",
            "detail": "iteration cap reached without a successful summarize call",
            "suggestion": "Call summarize earlier — once you have 4+ fetched documents, summarize immediately.",
        })

    if run_log.get("status") == "partial_llm_error":
        issues.append({
            "tag": "llm_error",
            "detail": run_log.get("llm_error", "unknown LLM error"),
            "suggestion": "Verify ANTHROPIC_API_KEY is set before invoking the agent.",
        })

    return issues


# ---------- Layer 3: LLM judge (Haiku) ----------

JUDGE_PROMPT = """You are scoring an intelligence briefing produced by an agent for a RapidSOS operator.

User prompt: {prompt}

Briefing:
{briefing}

Return STRICT JSON only (no markdown fence, no commentary):
{{
  "usefulness": 1-5 integer,
  "missing": ["up to 3 things a RapidSOS operator would want but the briefing did not cover"],
  "next_run_suggestion": "ONE concrete, actionable suggestion for the agent's next run on a similar prompt. Start with an imperative verb. Max 25 words."
}}
"""


def llm_judge(prompt: str, briefing: str) -> dict:
    """Single Haiku call. Returns judge dict or {error: ...} on failure."""
    try:
        response = llm.call_with_retry({
            "model": HAIKU_MODEL,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": JUDGE_PROMPT.format(prompt=prompt, briefing=briefing[:8000]),
            }],
        })
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    # Strip accidental ```json fences if Haiku adds them despite the instruction.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": f"judge returned non-JSON: {exc}", "raw": text[:200]}


# ---------- Composer + persistence ----------

def run_self_eval(run_log: dict, briefing_text: str, *, use_llm_judge: bool = True) -> dict:
    """Compose all three layers into one eval record."""
    eval_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": run_log.get("prompt"),
        "briefing_path": run_log.get("briefing_path"),
        "structure": check_structure(briefing_text),
        "trace_issues": critique_trace(run_log),
    }
    if use_llm_judge and os.environ.get("ANTHROPIC_API_KEY"):
        eval_record["judge"] = llm_judge(run_log.get("prompt", ""), briefing_text)
    else:
        eval_record["judge"] = {"skipped": True}
    return eval_record


def save_eval(eval_record: dict) -> None:
    """Append one JSON line to data/logs/evals.jsonl."""
    path = storage.LOGS / "evals.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")


def load_recent_lessons(n: int = MAX_LESSONS) -> list[str]:
    """Tail evals.jsonl, return the last N concrete next-run suggestions.

    Combines `trace_issues[*].suggestion` (deterministic, high-confidence) with
    `judge.next_run_suggestion` (LLM, broader). Deduplicates by suggestion text.
    """
    path = storage.LOGS / "evals.jsonl"
    if not path.exists():
        return []

    lessons: list[str] = []
    seen: set[str] = set()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    for line in reversed(lines):  # newest first
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidates = [i.get("suggestion") for i in rec.get("trace_issues", []) if i.get("suggestion")]
        judge = rec.get("judge") or {}
        if isinstance(judge, dict) and judge.get("next_run_suggestion"):
            candidates.append(judge["next_run_suggestion"])
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                lessons.append(c)
                if len(lessons) >= n:
                    return lessons
    return lessons


def lessons_block(lessons: list[str]) -> str:
    """Format lessons as a markdown block to append to SYSTEM_PROMPT.

    DEPRECATED — kept for backward compat. Use `demonstrations_block()` instead
    (U3). Lessons-as-rules don't transfer reliably per the strict eval; the
    successor mechanism is few-shot demonstrations sourced from kept/reverted
    runs scored under the composite verdict.

    Empty list → empty string (no-op append).
    """
    if not lessons:
        return ""
    bullets = "\n".join(f"- {lesson}" for lesson in lessons)
    return (
        "\n\n## Recent lessons (from prior runs — apply them)\n"
        f"{bullets}\n"
    )


# ============================================================
# U3: few-shot demonstrations replacing abstract lessons
# ============================================================

# Marker for edit_history records produced under the new composite verdict.
# Records without this field (or with structure-v0) are excluded from
# demonstrations to prevent teaching old-metric biases.
COMPOSITE_METRIC_VERSION = "composite-v1"
GOAL_METRIC_VERSION = "goal-v1"  # U4 — tag for goal-search output runs
ACCEPTED_METRIC_VERSIONS = (COMPOSITE_METRIC_VERSION, GOAL_METRIC_VERSION)
MIN_COMPOSITE_RECORDS = 3
MAX_DEMO_BRIEFING_CHARS = 200
MAX_TIMESTAMP_WINDOW_HOURS = 24


def _parse_iso(ts: str):
    """Parse ISO 8601 timestamp, returning None on failure."""
    if not ts:
        return None
    try:
        # Normalize trailing Z to +00:00 for fromisoformat compat
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _find_matching_run(edit_record: dict, runs: list[dict]) -> dict | None:
    """Find the most recent run in `runs` whose timestamp is BEFORE the edit
    record's timestamp, within MAX_TIMESTAMP_WINDOW_HOURS. This is the run
    whose eval triggered the edit proposal.
    """
    edit_ts = _parse_iso(edit_record.get("timestamp", ""))
    if edit_ts is None:
        return None
    window = timedelta(hours=MAX_TIMESTAMP_WINDOW_HOURS)
    best = None
    for r in runs:
        run_ts = _parse_iso(r.get("timestamp", ""))
        if run_ts is None or run_ts >= edit_ts:
            continue
        if (edit_ts - run_ts) > window:
            continue
        if best is None or run_ts > _parse_iso(best["timestamp"]):
            best = r
    return best


def _format_tool_trace(tool_calls: list[dict]) -> str:
    """Compact one-line summary of the tool-call sequence."""
    counts = {}
    errors = 0
    for c in tool_calls or []:
        counts[c.get("name", "?")] = counts.get(c.get("name", "?"), 0) + 1
        if c.get("error"):
            errors += 1
    parts = []
    for name in ("search", "fetch", "summarize"):
        if counts.get(name):
            parts.append(f"{counts[name]} {name}")
    if errors:
        parts.append(f"{errors} error(s)")
    return " · ".join(parts) if parts else "(no tool calls)"


def load_few_shot_examples(
    n_good: int = 2,
    n_bad: int = 1,
    min_records: int = MIN_COMPOSITE_RECORDS,
) -> dict:
    """Load few-shot demonstrations from edit_history.jsonl + runs.jsonl.

    Returns:
      {
        "ready": bool,                 # True iff >= min_records composite scored
        "good": [example_dict, ...],   # kept=True, sorted by composite desc
        "bad": [example_dict, ...],    # kept=False, sorted by composite asc
        "n_composite_records": int,
      }

    Demonstrations are GATED behind `min_records` composite-v1 edit_history
    records (adversarial F3 mitigation — pre-composite labels would teach the
    old metric's biases).
    """
    edit_history_path = storage.LOGS / "edit_history.jsonl"
    runs_path = storage.LOGS / "runs.jsonl"

    if not edit_history_path.exists():
        return {"ready": False, "good": [], "bad": [], "n_composite_records": 0}

    edits = []
    for line in edit_history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            edits.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Filter to composite-scored records only
    composite_edits = [
        e for e in edits if e.get("metric_version") in ACCEPTED_METRIC_VERSIONS
    ]
    n = len(composite_edits)

    if n < min_records:
        return {"ready": False, "good": [], "bad": [], "n_composite_records": n}

    # Load runs.jsonl for timestamp matching
    runs = []
    if runs_path.exists():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _build_example(edit: dict) -> dict | None:
        run = _find_matching_run(edit, runs)
        if run is None:
            return None
        # Load briefing snippet if path exists
        briefing_snippet = ""
        briefing_rel = run.get("briefing_path", "")
        if briefing_rel:
            bp = Path(briefing_rel)
            if not bp.is_absolute():
                bp = storage.ROOT / briefing_rel
            if bp.exists():
                text = bp.read_text(encoding="utf-8", errors="replace")
                briefing_snippet = text[:MAX_DEMO_BRIEFING_CHARS].strip()
        return {
            "edit_timestamp": edit.get("timestamp"),
            "composite_score": edit.get("composite_score") or edit.get("canary_delta"),
            "breakdown": edit.get("composite_breakdown"),
            "prompt": (run.get("prompt") or "")[:100],
            "trace": _format_tool_trace(run.get("tool_calls") or []),
            "briefing_snippet": briefing_snippet,
        }

    good_edits = [e for e in composite_edits if e.get("kept") is True]
    bad_edits = [e for e in composite_edits if e.get("kept") is False]

    # Sort by composite_score: good descending (highest-scoring exemplars
    # first), bad ascending (worst regressions first). The docstring claims
    # this; previously the slice took whatever order was on disk. (Review M-09.)
    good_edits.sort(key=lambda e: e.get("composite_score") or 0.0, reverse=True)
    bad_edits.sort(key=lambda e: e.get("composite_score") or 1.0)

    good_examples = [ex for ex in (_build_example(e) for e in good_edits) if ex][:n_good]
    bad_examples = [ex for ex in (_build_example(e) for e in bad_edits) if ex][:n_bad]

    return {
        "ready": True,
        "good": good_examples,
        "bad": bad_examples,
        "n_composite_records": n,
    }


def demonstrations_block(examples: dict | None = None) -> str:
    """Render few-shot examples as a markdown block to append to SYSTEM_PROMPT.

    Replaces lessons_block — demonstrations carry more behavioral signal than
    abstract rules per the strict eval. Empty/not-ready → empty string.

    If `examples` is None, calls load_few_shot_examples() with defaults.
    """
    if examples is None:
        examples = load_few_shot_examples()

    if not examples.get("ready"):
        return ""

    good = examples.get("good") or []
    bad = examples.get("bad") or []
    if not good and not bad:
        return ""

    lines = ["\n\n## Recent examples (compounded from prior cycles)"]

    if good:
        lines.append("\n### Good runs (kept after canary)\n")
        for ex in good:
            score = ex.get("composite_score")
            score_str = f"score={score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(f"- [{ex.get('edit_timestamp', '?')[:10]} {score_str}]")
            lines.append(f"  Prompt: {ex.get('prompt', '')!r}")
            lines.append(f"  Trace: {ex.get('trace', '')}")
            if ex.get("briefing_snippet"):
                lines.append(f"  Briefing: {ex['briefing_snippet']!r}")

    if bad:
        lines.append("\n### Anti-patterns (canary discarded or reverted)\n")
        for ex in bad:
            score = ex.get("composite_score")
            score_str = f"score={score:.2f}" if isinstance(score, (int, float)) else ""
            lines.append(f"- [{ex.get('edit_timestamp', '?')[:10]} {score_str}]")
            lines.append(f"  Prompt: {ex.get('prompt', '')!r}")
            lines.append(f"  Trace: {ex.get('trace', '')}")
            if ex.get("briefing_snippet"):
                lines.append(f"  Briefing: {ex['briefing_snippet']!r}")

    return "\n".join(lines) + "\n"
