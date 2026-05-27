"""Self-evaluation after every run. Three layers:

1. Structure check — deterministic regex over the briefing markdown.
2. Trace critique — scan tool_calls for known anti-patterns from runs.jsonl.
3. LLM judge — one Haiku call for usefulness + next-run suggestion.

Output appended to `data/logs/evals.jsonl`. Lessons feed back into the
SYSTEM_PROMPT on subsequent runs via `load_recent_lessons()` — that is the
compounding loop: the agent reads its own critique and adjusts.
"""

import json
import os
import re
from datetime import datetime, timezone

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
            issues.append({
                "tag": "summarize_missing_documents",
                "detail": f"summarize called {len(bad_args)} time(s) without `documents` arg",
                "suggestion": "Call summarize with BOTH prompt and documents — pass the full list of fetched document dicts as `documents`.",
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

    Empty list → empty string (no-op append).
    """
    if not lessons:
        return ""
    bullets = "\n".join(f"- {lesson}" for lesson in lessons)
    return (
        "\n\n## Recent lessons (from prior runs — apply them)\n"
        f"{bullets}\n"
    )
