"""Meta-eval cycle: agent improves its own SYSTEM_PROMPT.

Three-input compounding: recent lessons (last 5), full lesson arc (all evals
bucketed by tag), and edit_history (what we've tried, what worked, what
regressed). Sonnet receives all three; its META_EVAL_PROMPT forbids
re-proposing anchors that previously regressed and encourages variations on
improvements.

Verbs:
- propose   — Sonnet emits structured JSON of proposed SYSTEM_PROMPT edits
- apply     — applies one edit (by id) to agent/prompts.py, snapshots first,
              opens an edit_history record
- compare   — re-runs a historical prompt, scores the delta, closes the
              verdict on the trailing edit_history record
- keep      — marks the trailing edit as kept=true (closes the loop)
- revert    — restores prompts.py from snapshot, marks kept=false
- status    — dashboard: trailing record + recent lessons
- autonomous — full hybrid cycle: propose → canary → act, with safety nets

See docs/plans/2026-05-27-001-feat-meta-eval-loop-plan.md for the full design.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from agent import evaluate, llm, storage

SONNET_MODEL = "claude-haiku-4-5-20251001"  # swapped to Haiku for budget; was claude-sonnet-4-6

# Single source of truth for the metric_version tag. Re-exported from
# evaluate.COMPOSITE_METRIC_VERSION so a future bump touches one constant.
# (Code review M-10/AC-4 — was hardcoded in 4 places.)
def _metric_version() -> str:
    from agent import evaluate as _eval
    return _eval.COMPOSITE_METRIC_VERSION


def _count_replay_errors(run_result: dict) -> int:
    """Count tool-call errors in a replay run. Used by run_canary to wire the
    errors component of the composite verdict (fixes the dead-code bug from
    review M-04/REL-003/C3 where the previous expression always evaluated to
    an empty list)."""
    # replay_run returns the same shape as loop.run; tool_calls live in the
    # run record on disk. The in-memory return dict carries tool_call_count
    # but not the full tool_calls list, so we approximate: any non-complete
    # status counts as 1 error, and partial_* statuses count higher. Replay
    # is too cheap to round-trip through disk just for an exact error count.
    status = run_result.get("status", "")
    if status == "complete":
        return 0
    if status == "partial_empty_response":
        return 2  # severe
    if status.startswith("partial_"):
        return 1
    return 1  # unknown non-complete status


# Lazy path accessors — must read storage.LOGS / storage.ROOT at call time, not
# import time, because the temp_data_dir fixture monkey-patches those globals
# AFTER this module is imported. Module-level constants would freeze the real
# data/logs/ path and bypass test isolation.

def _meta_evals_dir():
    return storage.LOGS / "meta_evals"

def _edit_history_path():
    return storage.LOGS / "edit_history.jsonl"

def _prompts_history_dir():
    return storage.LOGS / "prompts_history"

def _prompts_py_path():
    return storage.ROOT / "agent" / "prompts.py"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _ensure_dirs() -> None:
    _meta_evals_dir().mkdir(parents=True, exist_ok=True)
    _prompts_history_dir().mkdir(parents=True, exist_ok=True)


# ============================================================
# U1: propose() — three-input compounding meta-eval
# ============================================================

def load_eval_corpus() -> dict:
    """Read evals.jsonl, partition into recent (last 5) + arc (all, bucketed).

    Returns:
        {
          "recent": [eval_record, ...],          # last 5, verbatim
          "arc": {tag: {count, first_seen, last_seen}, ...},  # all-time
          "total_runs": int,
        }
    """
    path = storage.LOGS / "evals.jsonl"
    if not path.exists():
        return {"recent": [], "arc": {}, "total_runs": 0}

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    arc: dict[str, dict] = {}
    for rec in records:
        ts = rec.get("timestamp", "")
        for issue in rec.get("trace_issues") or []:
            tag = issue.get("tag", "unknown")
            bucket = arc.setdefault(tag, {"count": 0, "first_seen": ts, "last_seen": ts})
            bucket["count"] += 1
            if ts < bucket["first_seen"]:
                bucket["first_seen"] = ts
            if ts > bucket["last_seen"]:
                bucket["last_seen"] = ts

    return {
        "recent": records[-5:],
        "arc": arc,
        "total_runs": len(records),
    }


def load_edit_history() -> list[dict]:
    """Read edit_history.jsonl. Returns [] when missing or empty."""
    if not _edit_history_path().exists():
        return []
    out = []
    for line in _edit_history_path().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _partition_edit_history(history: list[dict]) -> dict:
    """Group edits by verdict for the meta-eval prompt's prior-attempts payload."""
    out = {"improved": [], "regressed": [], "inconclusive": [], "pending": [], "discarded": []}
    for h in history:
        verdict = h.get("verdict", "pending")
        # canary verdicts also map cleanly
        if verdict == "canary_promoted":
            verdict = "improved"
        elif verdict == "canary_discarded":
            verdict = "discarded"
        bucket = out.setdefault(verdict, [])
        bucket.append({
            "anchor": h.get("anchor"),
            "new_text": h.get("new_text", "")[:200],
            "rationale": h.get("rationale", "")[:200],
            "verdict": verdict,
            "kept": h.get("kept"),
        })
    return out


META_EVAL_PROMPT = """You are auditing the SYSTEM_PROMPT of an intelligence-briefing agent for RapidSOS. Your job is to propose CONCRETE, ANCHORED edits to that SYSTEM_PROMPT based on observed run failures.

# Current SYSTEM_PROMPT

```
{system_prompt}
```

# Recent lessons (last 5 evals, newest first)

{recent_block}

# Full lesson arc (all-time pattern counts)

{arc_block}

# Edit history (what we've TRIED to fix and the outcome)

{history_block}

# Hard rules

1. **Do NOT propose any edit whose `anchor` substring matches an edit listed under "regressed" or "discarded" in the edit history.** Those exact fixes failed; re-proposing them wastes cycles.
2. **DO consider variations** on edits listed under "improved" — adjacent issues, tighter phrasing, expanded coverage.
3. **For patterns with `count >= 3` and zero successful edits in history**, propose a STRUCTURALLY different approach (e.g., add a tool-use rule rather than rephrase an existing one). Repeating the same shape that already failed is also wasted.
4. **Each `anchor` MUST be an exact substring of the current SYSTEM_PROMPT.** No paraphrasing. The system applies edits via literal find/replace.
5. **Each anchor must be unique** in the SYSTEM_PROMPT. If a string appears multiple times, expand it until unique.
6. **Edit types**: `add` inserts `new_text` after the anchor line; `replace` swaps the anchor for `new_text`; `delete` removes the anchor entirely.

# Output

Return STRICT JSON only — no markdown fence, no commentary outside the JSON:

```
{{
  "patterns_observed": [
    {{"tag": "...", "count": N, "trend": "chronic|new|resolved", "interpretation": "..."}}
  ],
  "prior_attempts_considered": [
    {{"edit_anchor": "...", "verdict": "improved|regressed|discarded|inconclusive", "decision": "skip|vary|build_on"}}
  ],
  "proposed_edits": [
    {{
      "id": "E1",
      "type": "add|replace|delete",
      "anchor": "<exact substring of current SYSTEM_PROMPT>",
      "new_text": "...",
      "rationale": "...",
      "addresses_pattern_tag": "..."
    }}
  ]
}}
```

If no actionable edits exist (the agent is already well-tuned), return `proposed_edits: []` with a non-empty `patterns_observed` explaining why.
"""


def _build_recent_block(recent: list[dict]) -> str:
    if not recent:
        return "(none — no eval records yet)"
    lines = []
    for r in recent:
        ts = r.get("timestamp", "?")
        prompt = (r.get("prompt") or "")[:80]
        s = r.get("structure", {})
        issues = r.get("trace_issues") or []
        judge = r.get("judge") or {}
        suggestion = judge.get("next_run_suggestion", "") if isinstance(judge, dict) else ""
        lines.append(
            f"- [{ts}] prompt={prompt!r} · score={s.get('score', '?')}/5 "
            f"· issues={[i.get('tag') for i in issues]} "
            f"· judge_suggestion={suggestion!r}"
        )
    return "\n".join(lines)


def _build_arc_block(arc: dict) -> str:
    if not arc:
        return "(none)"
    rows = sorted(arc.items(), key=lambda kv: -kv[1]["count"])
    return "\n".join(
        f"- {tag}: count={info['count']}, first_seen={info['first_seen']}, last_seen={info['last_seen']}"
        for tag, info in rows
    )


def _build_history_block(partitioned: dict) -> str:
    if not any(partitioned.values()):
        return "(none — no prior edits attempted)"
    out_lines = []
    for verdict_key in ("improved", "regressed", "discarded", "inconclusive", "pending"):
        items = partitioned.get(verdict_key, [])
        if not items:
            continue
        out_lines.append(f"\n## {verdict_key.upper()}")
        for item in items:
            out_lines.append(
                f"- anchor={item['anchor']!r}\n"
                f"  new_text={item['new_text']!r}\n"
                f"  rationale={item['rationale']!r}\n"
                f"  kept={item.get('kept')}"
            )
    return "\n".join(out_lines)


def propose_edits(*, use_llm: bool = True) -> dict:
    """Main entry for the `propose` verb.

    Returns the parsed meta-eval JSON (or {error: ...} on LLM/parse failure).
    Persists to data/logs/meta_evals/{ts}.json regardless of outcome.
    """
    _ensure_dirs()
    corpus = load_eval_corpus()
    history = load_edit_history()
    partitioned = _partition_edit_history(history)

    # Short-circuit if there's nothing to learn from
    if corpus["total_runs"] == 0 and not history:
        result = {
            "timestamp": _now_iso(),
            "patterns_observed": [],
            "prior_attempts_considered": [],
            "proposed_edits": [],
            "note": "no eval records or edit history — nothing to propose",
        }
        _save_meta_eval(result)
        return result

    if not use_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        result = {
            "timestamp": _now_iso(),
            "patterns_observed": [
                {"tag": tag, "count": info["count"], "trend": "unknown", "interpretation": "no-llm mode"}
                for tag, info in corpus["arc"].items()
            ],
            "prior_attempts_considered": [],
            "proposed_edits": [],
            "note": "no-llm mode — empty proposed_edits",
        }
        _save_meta_eval(result)
        return result

    # Read the current SYSTEM_PROMPT from the live module
    from agent import prompts as _prompts_mod
    system_prompt_text = _prompts_mod.SYSTEM_PROMPT

    user_message = META_EVAL_PROMPT.format(
        system_prompt=system_prompt_text,
        recent_block=_build_recent_block(corpus["recent"]),
        arc_block=_build_arc_block(corpus["arc"]),
        history_block=_build_history_block(partitioned),
    )

    try:
        response = llm.call_with_retry({
            "model": SONNET_MODEL,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": user_message}],
        })
    except Exception as exc:
        result = {"timestamp": _now_iso(), "error": f"{type(exc).__name__}: {exc}"}
        _save_meta_eval(result)
        return result

    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        result = {"timestamp": _now_iso(), "error": f"non-JSON response: {exc}", "raw": text[:500]}
        _save_meta_eval(result)
        return result

    parsed["timestamp"] = _now_iso()
    _validate_anchors(parsed, system_prompt_text)
    _save_meta_eval(parsed)
    return parsed


def _validate_anchors(parsed: dict, system_prompt: str) -> None:
    """Mutate `parsed`: tag each proposed edit with `anchor_valid` and
    `anchor_count` (how many times the anchor appears in SYSTEM_PROMPT).

    Edits with anchor_count != 1 cannot be applied automatically; downstream
    `apply` will refuse them.
    """
    for edit in parsed.get("proposed_edits") or []:
        anchor = edit.get("anchor", "")
        count = system_prompt.count(anchor) if anchor else 0
        edit["anchor_count"] = count
        edit["anchor_valid"] = count == 1


def _save_meta_eval(parsed: dict) -> Path:
    """Persist meta-eval result. Filename uses slug timestamp; latest symlink
    or 'last.json' file is created for easy CLI use."""
    _ensure_dirs()
    path = _meta_evals_dir() / f"{_now_slug()}.json"
    path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    last_path = _meta_evals_dir() / "last.json"
    last_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_last_meta_eval() -> dict | None:
    """Convenience: return the most recent saved meta-eval, or None."""
    last_path = _meta_evals_dir() / "last.json"
    if not last_path.exists():
        return None
    try:
        return json.loads(last_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ============================================================
# U2: apply() — mutate prompts.py with snapshot + edit_history
# ============================================================

def _apply_edit_in_memory(edit: dict, baseline_text: str) -> str:
    """Pure string transformation. Returns the modified text.

    Raises ValueError if anchor is missing or not unique (defensive — caller
    should validate first via the anchor_valid flag from propose_edits).
    """
    anchor = edit.get("anchor", "")
    new_text = edit.get("new_text", "")
    edit_type = edit.get("type", "add")

    if not anchor:
        raise ValueError("edit missing anchor")
    count = baseline_text.count(anchor)
    if count == 0:
        raise ValueError(f"anchor not found in baseline: {anchor!r}")
    if count > 1:
        raise ValueError(f"anchor not unique (appears {count} times): {anchor!r}")

    if edit_type == "replace":
        return baseline_text.replace(anchor, new_text, 1)
    if edit_type == "delete":
        return baseline_text.replace(anchor, "", 1)
    # default: add — insert new_text right after the anchor, with a newline
    # if the anchor doesn't already end at a line break boundary
    replacement = anchor + ("\n" + new_text if not new_text.startswith("\n") else new_text)
    return baseline_text.replace(anchor, replacement, 1)


def snapshot_prompts() -> Path:
    """Copy current agent/prompts.py to prompts_history/{ts}-pre-edit.py."""
    _ensure_dirs()
    src = _prompts_py_path()
    snap = _prompts_history_dir() / f"{_now_slug()}-pre-edit.py"
    snap.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return snap


def record_edit_attempt(record: dict) -> None:
    """Append one JSON line to edit_history.jsonl."""
    _ensure_dirs()
    with _edit_history_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _find_edit_by_id(edit_id: str) -> dict | None:
    meta = load_last_meta_eval()
    if not meta:
        return None
    for edit in meta.get("proposed_edits") or []:
        if edit.get("id") == edit_id:
            return edit
    return None


def _extract_system_prompt_string(prompts_py_text: str) -> tuple[str, int, int]:
    """Find the SYSTEM_PROMPT string literal in prompts.py and return its
    inner content + start/end byte offsets of the f-string body.

    Assumes the pattern: SYSTEM_PROMPT = f\"\"\"...\"\"\"
    Returns (inner_text, body_start_offset, body_end_offset).
    """
    match = re.search(
        r'SYSTEM_PROMPT\s*=\s*f"""(.*?)"""',
        prompts_py_text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("could not locate SYSTEM_PROMPT = f\"\"\"...\"\"\" in prompts.py")
    return match.group(1), match.start(1), match.end(1)


def apply_edit(edit_id: str, *, yes: bool = False) -> dict:
    """Apply one proposed edit (by id) to agent/prompts.py.

    Without `yes=True`, prints the unified diff and exits without mutating
    the file. With `yes=True`: snapshots, writes the edit, appends a
    `verdict=pending` record to edit_history.jsonl.

    Returns {applied: bool, diff: str, snapshot_path: str|None, record: dict|None,
             error: str|None}.
    """
    edit = _find_edit_by_id(edit_id)
    if edit is None:
        return {"applied": False, "error": f"edit_id {edit_id} not found in last meta-eval"}
    if not edit.get("anchor_valid"):
        return {
            "applied": False,
            "error": f"anchor not unique (count={edit.get('anchor_count', '?')}); cannot apply automatically",
        }

    src_path = _prompts_py_path()
    src_text = src_path.read_text(encoding="utf-8")

    try:
        inner, body_start, body_end = _extract_system_prompt_string(src_text)
        new_inner = _apply_edit_in_memory(edit, inner)
    except ValueError as exc:
        return {"applied": False, "error": f"edit application failed: {exc}"}

    new_src_text = src_text[:body_start] + new_inner + src_text[body_end:]

    # Syntax check before commit
    try:
        compile(new_src_text, "agent/prompts.py", "exec")
    except SyntaxError as exc:
        return {"applied": False, "error": f"edit would break prompts.py syntax: {exc}"}

    import difflib
    diff = "".join(difflib.unified_diff(
        src_text.splitlines(keepends=True),
        new_src_text.splitlines(keepends=True),
        fromfile="agent/prompts.py (current)",
        tofile="agent/prompts.py (proposed)",
        n=3,
    ))

    if not yes:
        return {"applied": False, "diff": diff, "dry_run": True}

    snapshot = snapshot_prompts()
    src_path.write_text(new_src_text, encoding="utf-8")

    record = {
        "timestamp": _now_iso(),
        "edit_id": edit_id,
        "type": edit.get("type"),
        "anchor": edit.get("anchor"),
        "new_text": edit.get("new_text"),
        "rationale": edit.get("rationale"),
        "addresses_pattern_tag": edit.get("addresses_pattern_tag"),
        "snapshot_path": str(snapshot.relative_to(storage.ROOT)),
        "verdict": "pending",
        "before_score": None,
        "after_score": None,
        "kept": None,
        "metric_version": _metric_version(),  # U3: tags this record for demonstrations eligibility
    }
    record_edit_attempt(record)

    return {
        "applied": True,
        "diff": diff,
        "snapshot_path": str(snapshot),
        "record": record,
    }


def revert_last_edit(*, note: str | None = None) -> dict:
    """Restore prompts.py from the most recent snapshot. Mark trailing
    edit_history record kept=false."""
    history = load_edit_history()
    if not history:
        return {"reverted": False, "error": "no edit history to revert"}
    trailing = history[-1]
    if trailing.get("kept") is not None:
        return {
            "reverted": False,
            "error": f"trailing record already has kept={trailing['kept']!r} — refusing to corrupt history",
        }

    snapshot_path = storage.ROOT / trailing["snapshot_path"]
    if not snapshot_path.exists():
        return {"reverted": False, "error": f"snapshot file missing: {snapshot_path}"}

    _prompts_py_path().write_text(snapshot_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Update trailing record in-place by rewriting the file
    trailing["kept"] = False
    if note:
        trailing["note"] = note
    _rewrite_edit_history(history)

    return {"reverted": True, "snapshot_path": str(snapshot_path)}


def _rewrite_edit_history(records: list[dict]) -> None:
    """Replace edit_history.jsonl with the given records list."""
    _ensure_dirs()
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else "")
    _edit_history_path().write_text(text, encoding="utf-8")


def keep_last_edit(*, note: str | None = None) -> dict:
    """Mark trailing edit_history record kept=true."""
    history = load_edit_history()
    if not history:
        return {"kept": False, "error": "no edit history"}
    trailing = history[-1]
    if trailing.get("kept") is not None:
        return {
            "kept": False,
            "error": f"trailing record already has kept={trailing['kept']!r}",
        }
    trailing["kept"] = True
    if note:
        trailing["note"] = note
    _rewrite_edit_history(history)
    return {"kept": True, "record": trailing}


# ============================================================
# U7: canary — outcome-gated autonomy via full-loop replay
# ============================================================

def _canary_dir():
    return storage.LOGS / "canary"


def _select_canary_prompt() -> tuple[str | None, list[str]]:
    """Pick a historical prompt for canary use + its cached URL set.

    Strategy: find the most recent COMPLETE run in runs.jsonl whose briefing
    cites ≥2 URLs we have in the fetch cache. Falls back to filename-slug
    glob if the exact briefing_path from the log is missing (briefings may
    have been renamed). Returns (prompt, urls) or (None, []) if nothing
    usable.
    """
    runs_path = storage.LOGS / "runs.jsonl"
    if not runs_path.exists():
        return None, []
    runs = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    url_pattern = re.compile(r"\[\^(\d+)\]:\s*\[[^\]]+\]\((https?://[^)\s]+)\)")
    briefings_dir = storage.BRIEFINGS
    for run in reversed(runs):
        if run.get("status") != "complete":
            continue
        briefing_rel = run.get("briefing_path") or ""
        # Try exact path first
        briefing_path = Path(briefing_rel)
        if not briefing_path.is_absolute():
            briefing_path = storage.ROOT / briefing_rel
        if not briefing_path.exists():
            # Glob fallback: match by timestamp prefix (first 15 chars of slug)
            stem = Path(briefing_rel).stem
            if stem:
                prefix = stem[:15]  # YYYYMMDD-HHMMSS
                candidates = sorted(briefings_dir.glob(f"{prefix}*.md"))
                if candidates:
                    briefing_path = candidates[0]
                else:
                    continue
            else:
                continue
        text = briefing_path.read_text(encoding="utf-8", errors="replace")
        urls = list(dict.fromkeys(m.group(2) for m in url_pattern.finditer(text)))
        cached_urls = [u for u in urls if storage.load_fetched(u) is not None]
        if len(cached_urls) >= 2:
            return run["prompt"], cached_urls
    return None, []


def run_canary(edit: dict, *, k: int = 1, canary_prompt: str | None = None,
               canary_urls: list[str] | None = None) -> dict:
    """Run K canary mini-replays comparing baseline vs candidate SYSTEM_PROMPT.

    Args:
        edit: a proposed edit dict (from propose_edits)
        k: number of replay pairs (default 1 for cost; bump for production)
        canary_prompt: historical prompt to replay; auto-selected if None
        canary_urls: cached URLs to constrain replay; auto-selected if None

    Returns (current shape post-U5 composite migration):
        {
          "edit_id": str,
          "edit_type": str,
          "decision": "promote|discard|gate",
          "reason": str,
          "delta": float,                       # avg composite_delta across K
          "baseline_composite": float,          # avg baseline composite across K
          "candidate_composite": float,         # avg candidate composite across K
          "k": int,
          "k_results": [
            {
              "iteration": int,
              "canary_prompt": str,
              "baseline_composite": float,
              "candidate_composite": float,
              "composite_delta": float,
              "baseline_breakdown": {operator, structure, efficiency, errors},
              "candidate_breakdown": {operator, structure, efficiency, errors},
              "baseline_status": str, "candidate_status": str,
              "baseline_tokens": dict, "candidate_tokens": dict,
              "baseline_op_tokens": dict, "candidate_op_tokens": dict,
            },
            ...
          ],
          "metric_version": "composite-v1",
        }
    """
    from agent import evaluate as _eval, replay as _replay, prompts as _prompts_mod

    _canary_dir().mkdir(parents=True, exist_ok=True)

    # 1. Hard-override: type=delete always discards regardless of canary
    if edit.get("type") == "delete":
        result = {
            "edit_id": edit.get("id"),
            "decision": "discard",
            "reason": "type=delete: blast radius too high for canary auto-promote",
            "k_results": [],
            "delta": None,
        }
        _save_canary(result)
        return result

    # 2. Build candidate SYSTEM_PROMPT in memory
    baseline_text = _prompts_mod.SYSTEM_PROMPT
    try:
        candidate_text = _apply_edit_in_memory(edit, baseline_text)
    except ValueError as exc:
        result = {
            "edit_id": edit.get("id"),
            "decision": "discard",
            "reason": f"edit-application failed: {exc}",
            "k_results": [],
            "delta": None,
        }
        _save_canary(result)
        return result

    # 3. Pick canary prompt + URLs if not provided
    if canary_prompt is None or canary_urls is None:
        sel_prompt, sel_urls = _select_canary_prompt()
        canary_prompt = canary_prompt or sel_prompt
        canary_urls = canary_urls or sel_urls

    if not canary_prompt or len(canary_urls) < 2:
        result = {
            "edit_id": edit.get("id"),
            "decision": "gate",
            "reason": "no historical prompt with ≥2 cached URLs available — human review needed",
            "k_results": [],
            "delta": None,
        }
        _save_canary(result)
        return result

    # 4. K replay pairs — now using composite verdict (operator + structure + tokens + errors)
    from agent import operator_sim as _opsim
    k_results = []
    for i in range(k):
        baseline_run = _replay.replay_run(canary_prompt, canary_urls, system_override=baseline_text)
        candidate_run = _replay.replay_run(canary_prompt, canary_urls, system_override=candidate_text)

        baseline_brief = baseline_run.get("briefing", "")
        candidate_brief = candidate_run.get("briefing", "")

        # Structure (existing — used for both shapes; cheap to compute always)
        baseline_struct = _eval.check_structure(baseline_brief)
        candidate_struct = _eval.check_structure(candidate_brief)

        # U4 — output-shape dispatch. Detect on baseline; apply the same
        # scorer to candidate to keep the verdict apples-to-apples.
        # candidates_list output → goal_coverage_score; briefing → coverage_score.
        baseline_shape = _opsim.detect_output_shape(baseline_brief)
        if baseline_shape == "candidates":
            baseline_op = _opsim.goal_coverage_score(canary_prompt, baseline_brief)
            candidate_op = _opsim.goal_coverage_score(canary_prompt, candidate_brief)
        else:
            baseline_op = _opsim.coverage_score(canary_prompt, baseline_brief)
            candidate_op = _opsim.coverage_score(canary_prompt, candidate_brief)

        # Trace issues from each replay run. Fixes the dead-code bug flagged
        # by 5 reviewers: the previous expression `[... for c in (x and [] or [])]`
        # always evaluated to [] regardless of x, so the errors component of
        # the composite was permanently 1.0 (silently dead 10% weight).
        # Now we surface real summarize/fetch failures from the replay traces.
        baseline_errors = _count_replay_errors(baseline_run)
        candidate_errors = _count_replay_errors(candidate_run)

        # Build minimal eval_records for composite scoring. trace_issues is a
        # list whose LENGTH is what composite_score consumes (n_errors), so a
        # list of N placeholder dicts is the right shape.
        baseline_eval = {
            "operator_sim": baseline_op,
            "structure": baseline_struct,
            "tokens": baseline_run.get("tokens"),
            "trace_issues": [{"tag": "replay_error"} for _ in range(baseline_errors)],
        }
        candidate_eval = {
            "operator_sim": candidate_op,
            "structure": candidate_struct,
            "tokens": candidate_run.get("tokens"),
            "trace_issues": [{"tag": "replay_error"} for _ in range(candidate_errors)],
        }

        baseline_composite = _opsim.composite_score(baseline_eval)
        candidate_composite = _opsim.composite_score(candidate_eval)

        k_results.append({
            "iteration": i,
            "canary_prompt": canary_prompt,
            "baseline_composite": baseline_composite["composite"],
            "candidate_composite": candidate_composite["composite"],
            "composite_delta": round(
                candidate_composite["composite"] - baseline_composite["composite"], 4
            ),
            "baseline_breakdown": baseline_composite["breakdown"],
            "candidate_breakdown": candidate_composite["breakdown"],
            "baseline_status": baseline_run.get("status"),
            "candidate_status": candidate_run.get("status"),
            "baseline_tokens": baseline_run.get("tokens"),
            "candidate_tokens": candidate_run.get("tokens"),
            # Operator-sim Haiku tokens (review PERF-001/REL-004 fix). Without
            # these, autonomous() cost-cap accounting silently undercounts by
            # ~$0.01 per canary iteration; at K>=3 and multi-cycle runs that
            # accumulates past the budget threshold.
            "baseline_op_tokens": baseline_op.get("tokens"),
            "candidate_op_tokens": candidate_op.get("tokens"),
        })

    # 5. Classify via composite delta (averaged across K)
    decision, reason = classify_canary(edit, k_results)
    avg_delta = sum(r["composite_delta"] for r in k_results) / len(k_results)
    avg_baseline = sum(r["baseline_composite"] for r in k_results) / len(k_results)
    avg_candidate = sum(r["candidate_composite"] for r in k_results) / len(k_results)

    # U4 — pick metric_version based on detected output shape on the last
    # baseline replay. Goal-search runs get "goal-v1"; briefings get
    # "composite-v1". Both flow through the same demonstrations gate
    # (load_few_shot_examples widened to accept both).
    from agent import evaluate as _eval_mod
    last_shape = (
        _opsim.detect_output_shape(baseline_brief)
        if k_results else "unknown"
    )
    if last_shape == "candidates":
        metric_version_tag = _eval_mod.GOAL_METRIC_VERSION
    else:
        metric_version_tag = _eval_mod.COMPOSITE_METRIC_VERSION

    result = {
        "edit_id": edit.get("id"),
        "edit_type": edit.get("type"),
        "decision": decision,
        "reason": reason,
        "delta": round(avg_delta, 4),
        "baseline_composite": round(avg_baseline, 4),
        "candidate_composite": round(avg_candidate, 4),
        "k": k,
        "k_results": k_results,
        "metric_version": metric_version_tag,
    }
    _save_canary(result)
    return result


def classify_canary(edit: dict, k_results: list[dict]) -> tuple[str, str]:
    """Deterministic canary verdict via composite score deltas.

    Rules (REPLACED in rev2 — was integer-delta structure-only):
    - type=delete → always discard (high blast radius override)
    - empty k_results → gate
    - any composite_delta < DISCARD_DELTA → discard (severe regression)
    - average composite_delta > PROMOTE_DELTA → promote
    - otherwise → gate

    See agent.operator_sim for PROMOTE_DELTA / DISCARD_DELTA constants.
    """
    from agent import operator_sim as _opsim

    if edit.get("type") == "delete":
        return "discard", "type=delete: blast radius override"
    if not k_results:
        return "gate", "no canary results — empty K"

    deltas = [r["composite_delta"] for r in k_results]
    if any(d < _opsim.DISCARD_DELTA for d in deltas):
        return "discard", f"severe composite regression (min delta={min(deltas):+.3f})"
    avg_delta = sum(deltas) / len(deltas)
    if avg_delta > _opsim.PROMOTE_DELTA:
        return "promote", f"avg composite improved by {avg_delta:+.3f} across K={len(deltas)}"
    return "gate", f"composite delta {avg_delta:+.3f} within noise band — human review needed"


def _save_canary(result: dict) -> Path:
    result = {"timestamp": _now_iso(), **result}
    _canary_dir().mkdir(parents=True, exist_ok=True)
    eid = result.get("edit_id") or "noid"
    path = _canary_dir() / f"{_now_slug()}-{eid}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ============================================================
# U8: autonomous() — full hybrid cycle with safety nets
# ============================================================

# Anthropic Haiku 4.5 pricing (rough, for budget estimation only — not exact billing).
# Numbers in USD per 1M tokens. Update if Anthropic changes pricing.
# Swapped from Sonnet 4.6 ($3/$15 per 1M) to Haiku for $1.95 demo budget.
_SONNET_INPUT_PER_1M = 1.00
_SONNET_OUTPUT_PER_1M = 5.00


def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1_000_000) * _SONNET_INPUT_PER_1M + (tokens_out / 1_000_000) * _SONNET_OUTPUT_PER_1M


def _autonomous_runs_dir():
    return storage.LOGS / "autonomous_runs"


def autonomous(*, max_cycles: int = 3, max_cost: float = 0.50,
               canary_prompt: str | None = None, canary_urls: list[str] | None = None,
               canary_k: int = 1, dry_run: bool = False) -> dict:
    """Hybrid autonomous meta-eval cycle.

    Per-cycle flow:
      1. propose (Sonnet) — three-input compounding
      2. pick top valid edit from proposed_edits (by addresses_pattern_tag relevance)
      3. anchor reproposal guard: halt if anchor is in edit_history.regressed/discarded
      4. canary (U7) → promote/discard/gate
      5. act:
           promote → snapshot + apply --yes + record(verdict=canary_promoted, kept=True)
           discard → record skipped attempt (verdict=canary_discarded, kept=False)
           gate    → halt + surface for human
      6. update cumulative state, check safety nets, loop

    Safety nets (any trip → halt):
      - total_cost > max_cost
      - cycle_count >= max_cycles
      - 2 consecutive in-session regressions (canary_discarded)
      - convergence: propose returned zero proposed_edits
      - anchor-reproposal: meta-eval ignored its own discipline rule

    Returns a summary dict; full audit log written to data/logs/autonomous_runs/.
    """
    _ensure_dirs()
    _autonomous_runs_dir().mkdir(parents=True, exist_ok=True)

    session_id = _now_slug()
    audit_path = _autonomous_runs_dir() / f"{session_id}-autonomous.md"
    cycles: list[dict] = []
    total_cost = 0.0
    consecutive_regressions = 0
    halt_reason: str | None = None

    for cycle_idx in range(1, max_cycles + 1):
        cycle = {"cycle": cycle_idx, "events": []}

        # Pre-cycle safety nets
        if total_cost >= max_cost:
            halt_reason = f"cost_cap: ${total_cost:.4f} >= ${max_cost}"
            cycle["events"].append({"halt": halt_reason})
            cycles.append(cycle)
            break

        # 1. propose
        cycle["events"].append({"step": "propose"})
        proposal = propose_edits(use_llm=not dry_run)
        cycle["proposal"] = {
            "patterns": proposal.get("patterns_observed", []),
            "n_edits": len(proposal.get("proposed_edits", [])),
            "error": proposal.get("error"),
        }
        if proposal.get("error"):
            halt_reason = f"propose_failed: {proposal['error']}"
            cycle["events"].append({"halt": halt_reason})
            cycles.append(cycle)
            break

        # Convergence check
        valid_edits = [e for e in (proposal.get("proposed_edits") or []) if e.get("anchor_valid")]
        if not valid_edits:
            halt_reason = "convergence_no_valid_edits"
            cycle["events"].append({"halt": halt_reason})
            cycles.append(cycle)
            break

        # 2. Pick top edit (just take the first valid one — proposer is expected to
        # rank by relevance; further heuristics deferred)
        edit = valid_edits[0]
        cycle["chosen_edit"] = {
            "id": edit.get("id"),
            "type": edit.get("type"),
            "anchor": edit.get("anchor"),
            "new_text": edit.get("new_text", "")[:200],
            "rationale": edit.get("rationale", ""),
        }

        # 3. Anchor reproposal guard
        history = load_edit_history()
        bad_anchors = {
            h.get("anchor") for h in history
            if h.get("verdict") in ("regressed", "canary_discarded") or h.get("kept") is False
        }
        if edit.get("anchor") in bad_anchors:
            halt_reason = f"anchor_reproposal: '{edit.get('anchor')}' was previously discarded/reverted"
            cycle["events"].append({"halt": halt_reason})
            cycles.append(cycle)
            break

        # 4. canary
        cycle["events"].append({"step": "canary"})
        canary_result = run_canary(edit, k=canary_k,
                                    canary_prompt=canary_prompt,
                                    canary_urls=canary_urls)
        cycle["canary"] = {
            "decision": canary_result["decision"],
            "reason": canary_result["reason"],
            "delta": canary_result.get("delta"),
        }
        # Accumulate canary cost. Includes BOTH the full-loop replay tokens
        # AND the operator_sim Haiku tokens (per code review PERF-001/REL-004
        # — previously the op_tokens were invisible to the cost cap).
        for kr in canary_result.get("k_results") or []:
            for which in (
                "baseline_tokens", "candidate_tokens",
                "baseline_op_tokens", "candidate_op_tokens",
            ):
                tk = kr.get(which) or {}
                total_cost += _estimate_cost(tk.get("input", 0), tk.get("output", 0))

        # 5. act
        if canary_result["decision"] == "promote":
            apply_result = apply_edit(edit["id"], yes=not dry_run) if not dry_run else {"applied": False, "dry_run": True}
            cycle["apply"] = {"applied": apply_result.get("applied"), "error": apply_result.get("error")}
            if apply_result.get("applied"):
                # Update trailing edit_history record verdict to canary_promoted + kept=true
                hist = load_edit_history()
                if hist:
                    hist[-1]["verdict"] = "canary_promoted"
                    hist[-1]["kept"] = True
                    hist[-1]["canary_delta"] = canary_result.get("delta")
                    hist[-1]["composite_score"] = canary_result.get("candidate_composite")
                    hist[-1]["composite_breakdown"] = (
                        canary_result.get("k_results") or [{}]
                    )[0].get("candidate_breakdown")
                    hist[-1]["metric_version"] = _metric_version()
                    _rewrite_edit_history(hist)
                consecutive_regressions = 0
                cycle["events"].append({"result": "promoted"})
            else:
                # Apply failed (anchor conflict after prior edits, etc.) — treat as gate
                halt_reason = f"apply_failed: {apply_result.get('error')}"
                cycle["events"].append({"halt": halt_reason})
                cycles.append(cycle)
                break

        elif canary_result["decision"] == "discard":
            # Record the attempt as canary_discarded so it shows up in edit_history
            record_edit_attempt({
                "timestamp": _now_iso(),
                "edit_id": edit.get("id"),
                "type": edit.get("type"),
                "anchor": edit.get("anchor"),
                "new_text": edit.get("new_text"),
                "rationale": edit.get("rationale"),
                "addresses_pattern_tag": edit.get("addresses_pattern_tag"),
                "snapshot_path": None,  # never applied
                "verdict": "canary_discarded",
                "canary_delta": canary_result.get("delta"),
                "canary_reason": canary_result.get("reason"),
                "before_score": None, "after_score": None,
                "composite_score": canary_result.get("candidate_composite"),
                "composite_breakdown": (canary_result.get("k_results") or [{}])[0].get("candidate_breakdown"),
                "kept": False,
                "metric_version": _metric_version(),
            })
            consecutive_regressions += 1
            cycle["events"].append({"result": "discarded"})

        else:  # gate
            halt_reason = f"canary_gate: {canary_result['reason']}"
            cycle["events"].append({"halt": halt_reason})
            cycles.append(cycle)
            break

        # 6. Post-cycle safety nets
        cycles.append(cycle)
        if consecutive_regressions >= 2:
            halt_reason = "consecutive_regressions: 2 canary discards in a row"
            break

    if halt_reason is None:
        halt_reason = "max_cycles_hit" if len(cycles) >= max_cycles else "loop_exited"

    summary = {
        "session_id": session_id,
        "halt_reason": halt_reason,
        "cycles_run": len(cycles),
        "total_cost_estimate": round(total_cost, 4),
        "cycles": cycles,
    }
    _render_autonomous_audit(audit_path, summary)
    summary["audit_path"] = str(audit_path)
    return summary


def _render_autonomous_audit(path: Path, summary: dict) -> None:
    """Markdown audit log for one autonomous session."""
    lines = [
        f"# Autonomous meta-eval session — {summary['session_id']}",
        f"",
        f"**Halt reason:** `{summary['halt_reason']}`",
        f"**Cycles run:** {summary['cycles_run']}",
        f"**Estimated cost:** ${summary['total_cost_estimate']:.4f}",
        f"",
        "---",
        "",
    ]
    for cycle in summary["cycles"]:
        lines.append(f"## Cycle {cycle['cycle']}")
        prop = cycle.get("proposal", {})
        lines.append(f"- Patterns observed: {len(prop.get('patterns') or [])}")
        lines.append(f"- Proposed edits: {prop.get('n_edits')}")
        if prop.get("error"):
            lines.append(f"- Propose error: `{prop['error']}`")
        edit = cycle.get("chosen_edit")
        if edit:
            lines.append(f"- Chosen edit `{edit['id']}` ({edit['type']})")
            lines.append(f"  - Anchor: `{edit['anchor']}`")
            lines.append(f"  - Rationale: {edit['rationale']}")
        canary = cycle.get("canary")
        if canary:
            lines.append(f"- Canary: **{canary['decision']}** — {canary['reason']} (delta={canary['delta']})")
        for ev in cycle.get("events", []):
            lines.append(f"  - {ev}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# U3 (light): compare() — re-run a historical prompt and diff
# ============================================================

def compare(prompt_substring: str) -> dict:
    """Re-run the most recent matching historical prompt, score the delta vs
    its saved briefing, update trailing edit_history record verdict.

    Returns {prompt, before_score, after_score, verdict, report_path}.
    """
    from agent import evaluate as _eval, loop as _loop

    runs_path = storage.LOGS / "runs.jsonl"
    if not runs_path.exists():
        return {"error": "no runs.jsonl"}

    runs = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    matches = [r for r in runs if prompt_substring.lower() in (r.get("prompt") or "").lower()]
    if not matches:
        return {"error": f"no run matching {prompt_substring!r}"}
    historical = matches[-1]
    prompt = historical["prompt"]

    # Load before briefing
    before_path = historical.get("briefing_path", "")
    before_full = Path(before_path) if Path(before_path).is_absolute() else storage.ROOT / before_path
    before_brief = before_full.read_text(encoding="utf-8") if before_full.exists() else ""
    before_score = _eval.check_structure(before_brief)["score"] if before_brief else 0

    # Re-run agent with current SYSTEM_PROMPT
    after_result = _loop.run(prompt, self_eval=False)
    after_brief = after_result.get("briefing", "")
    after_score = _eval.check_structure(after_brief)["score"]

    # Heuristic verdict
    delta = after_score - before_score
    if delta > 0:
        verdict = "improved"
    elif delta < 0:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    # Update trailing edit_history record if pending
    history = load_edit_history()
    if history and history[-1].get("verdict") == "pending":
        history[-1]["before_score"] = before_score
        history[-1]["after_score"] = after_score
        history[-1]["verdict"] = verdict
        _rewrite_edit_history(history)

    # Render comparison report
    comparisons_dir = storage.LOGS / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    slug = storage.slugify(prompt, max_len=40)
    report_path = comparisons_dir / f"{_now_slug()}-{slug}.md"
    report = (
        f"# Comparison — {prompt[:80]}\n\n"
        f"**Before score:** {before_score}/5 · **After score:** {after_score}/5 · "
        f"**Verdict:** `{verdict}` (delta={delta:+d})\n\n"
        f"---\n\n## BEFORE\n\n{before_brief}\n\n---\n\n## AFTER\n\n{after_brief}\n"
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "prompt": prompt, "before_score": before_score, "after_score": after_score,
        "delta": delta, "verdict": verdict, "report_path": str(report_path),
    }


# ============================================================
# CLI entry — python -m agent.meta_eval <verb>
# ============================================================

def status() -> dict:
    """Print/return a dashboard: trailing edit + demonstrations state + last meta-eval.

    Reports both the legacy `recent_lessons` (for backward compat) AND the new
    `demonstrations_*` fields. The demonstrations layer is gated behind
    MIN_COMPOSITE_RECORDS records of metric_version composite-v1; surfacing
    readiness + count lets the user see why SYSTEM_PROMPT isn't getting demos
    yet during the bootstrap window. (Code review M-03 + agent-native gap.)
    """
    history = load_edit_history()
    trailing = history[-1] if history else None
    last_meta = load_last_meta_eval()
    from agent import evaluate as _eval
    lessons = _eval.load_recent_lessons()
    demos = _eval.load_few_shot_examples()
    return {
        "trailing_edit": trailing,
        "history_count": len(history),
        "last_meta_eval_ts": (last_meta or {}).get("timestamp"),
        "last_meta_eval_n_edits": len((last_meta or {}).get("proposed_edits") or []),
        "recent_lessons": lessons,  # legacy
        "demonstrations_ready": demos.get("ready", False),
        "n_composite_records": demos.get("n_composite_records", 0),
        "n_good_examples": len(demos.get("good") or []),
        "n_bad_examples": len(demos.get("bad") or []),
    }


_USAGE = """\
Usage: python -m agent.meta_eval <verb> [args]

Verbs:
  propose [--no-llm]            Run Sonnet meta-eval over evals.jsonl + edit_history.jsonl
                                → proposed SYSTEM_PROMPT edits. Saves to data/logs/meta_evals/.
  apply <edit_id> [--yes]       Apply one proposed edit to agent/prompts.py.
                                Without --yes: print diff, leave file unchanged.
  compare <prompt_substring>    Re-run a historical prompt with current SYSTEM_PROMPT;
                                score the delta + write a comparison report.
  keep [--note TEXT]            Mark trailing edit_history record kept=true.
  revert [--note TEXT]          Restore prompts.py from snapshot; mark kept=false.
  status                        Dashboard: trailing edit, demonstrations
                                readiness, last meta-eval.
  score <briefing_path>         Run operator_sim + composite_score on a single
                                briefing (offline debugging — no canary, no
                                replay). --prompt PROMPT for the user prompt
                                (defaults to the briefing's title line).
  autonomous [--max-cycles N] [--max-cost N] [--canary-k N]
                                Run hybrid cycle: propose → canary → act.
                                Halts on cost cap, cycle cap, consecutive regressions,
                                anchor reproposal, or convergence.
"""


def main(argv: list[str] | None = None) -> int:
    import sys
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        sys.stderr.write(_USAGE)
        return 2
    verb, rest = args[0], args[1:]

    if verb == "propose":
        use_llm = "--no-llm" not in rest
        result = propose_edits(use_llm=use_llm)
        if "error" in result:
            sys.stderr.write(f"propose failed: {result['error']}\n")
            print(json.dumps(result, indent=2))
            return 1
        print(json.dumps(result, indent=2))
        n = len(result.get("proposed_edits") or [])
        sys.stderr.write(f"\nproposed {n} edit(s); patterns: {len(result.get('patterns_observed') or [])}\n")
        return 0

    if verb == "apply":
        if not rest:
            sys.stderr.write("apply: missing <edit_id>\n")
            return 2
        edit_id = rest[0]
        yes = "--yes" in rest
        result = apply_edit(edit_id, yes=yes)
        if "diff" in result:
            print(result["diff"])
        if not result.get("applied"):
            if result.get("dry_run"):
                sys.stderr.write("DRY RUN — pass --yes to apply\n")
                return 0
            sys.stderr.write(f"apply failed: {result.get('error')}\n")
            return 1
        sys.stderr.write(f"applied. snapshot: {result['snapshot_path']}\n")
        return 0

    if verb == "compare":
        if not rest:
            sys.stderr.write("compare: missing <prompt_substring>\n")
            return 2
        result = compare(rest[0])
        if "error" in result:
            sys.stderr.write(f"compare failed: {result['error']}\n")
            return 1
        sys.stderr.write(
            f"\nbefore={result['before_score']}/5 → after={result['after_score']}/5 "
            f"({result['verdict']}, delta={result['delta']:+d})\n"
            f"report: {result['report_path']}\n"
        )
        return 0 if result["verdict"] != "regressed" else 1

    if verb in ("keep", "revert"):
        note = None
        if "--note" in rest:
            i = rest.index("--note")
            if i + 1 < len(rest):
                note = rest[i + 1]
        fn = keep_last_edit if verb == "keep" else revert_last_edit
        result = fn(note=note)
        print(json.dumps(result, indent=2))
        success_key = "kept" if verb == "keep" else "reverted"
        return 0 if result.get(success_key) else 1

    if verb == "status":
        result = status()
        print(json.dumps(result, indent=2, default=str))
        return 0

    if verb == "score":
        if not rest:
            sys.stderr.write("score: missing <briefing_path>\n")
            return 2
        briefing_path = Path(rest[0])
        if not briefing_path.exists():
            sys.stderr.write(f"score: briefing not found at {briefing_path}\n")
            return 2
        briefing_text = briefing_path.read_text(encoding="utf-8")
        prompt = None
        if "--prompt" in rest:
            i = rest.index("--prompt")
            if i + 1 < len(rest):
                prompt = rest[i + 1]
        if prompt is None:
            # Default: extract first H1 line if present, else first non-empty line
            for line in briefing_text.splitlines():
                line = line.strip()
                if line.startswith("# "):
                    prompt = line[2:].lstrip("Briefing — ")
                    break
                if line and not line.startswith("#"):
                    prompt = line[:120]
                    break
            if prompt is None:
                prompt = "(unknown prompt)"
        from agent import evaluate as _eval, operator_sim as _opsim
        op = _opsim.coverage_score(prompt, briefing_text)
        struct = _eval.check_structure(briefing_text)
        composite = _opsim.composite_score({
            "operator_sim": op,
            "structure": struct,
        })
        result = {
            "briefing_path": str(briefing_path),
            "prompt": prompt,
            "structure": struct,
            "operator_sim": op,
            "composite": composite,
        }
        print(json.dumps(result, indent=2, default=str))
        return 0

    if verb == "autonomous":
        kwargs = {}
        if "--max-cycles" in rest:
            kwargs["max_cycles"] = int(rest[rest.index("--max-cycles") + 1])
        if "--max-cost" in rest:
            kwargs["max_cost"] = float(rest[rest.index("--max-cost") + 1])
        if "--canary-k" in rest:
            kwargs["canary_k"] = int(rest[rest.index("--canary-k") + 1])
        if "--dry-run" in rest:
            kwargs["dry_run"] = True
        result = autonomous(**kwargs)
        sys.stderr.write(
            f"\nautonomous halt: {result['halt_reason']} · "
            f"cycles={result['cycles_run']} · cost≈${result['total_cost_estimate']:.4f}\n"
            f"audit: {result['audit_path']}\n"
        )
        # Pretty-print cycles to stdout
        print(json.dumps({
            "halt_reason": result["halt_reason"],
            "cycles_run": result["cycles_run"],
            "cost_estimate": result["total_cost_estimate"],
            "cycles": [{
                "cycle": c["cycle"],
                "chosen_edit": c.get("chosen_edit", {}).get("id") if c.get("chosen_edit") else None,
                "canary": c.get("canary"),
                "apply": c.get("apply"),
            } for c in result["cycles"]],
        }, indent=2))
        return 0

    sys.stderr.write(f"unknown verb: {verb}\n{_USAGE}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
