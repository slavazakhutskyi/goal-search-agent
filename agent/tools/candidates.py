"""Candidate-accumulation tools for goal-completion search.

Two tools the model invokes during a goal-search loop:

- `add_candidate(url, name, why_fits, score, axes?)` — record a candidate that
  matches the user's goal, with a single relevance score in [0, 1] and an
  optional per-dimension axes dict.
- `finalize()` — signal "goal met; emit the ranked candidates list."

Both tools are stateless. The candidates list itself lives in `loop.run`'s
local state — the loop's dispatcher reads the tool result and upserts the
candidate into that list. Keeping the tool functions pure means
`tests/test_candidates.py` validates schema + validation logic without
touching the loop.
"""

import math
from typing import Any


# ---------------------------------------------------------------------------
# add_candidate
# ---------------------------------------------------------------------------

ADD_CANDIDATE_TOOL_SCHEMA = {
    "name": "add_candidate",
    "description": (
        "Record a candidate that matches the user's goal. Call this once per "
        "candidate as you fetch and evaluate sources. Provide a single "
        "`score` in [0, 1] expressing how well this candidate fits the stated "
        "goal, and a short `why_fits` grounded in fetched content. Calling "
        "again with the same `url` UPSERTS — replaces the prior entry. Use "
        "`axes` only when the goal naturally decomposes (e.g., timezone_match, "
        "skill_fit, freshness) and you want to surface the breakdown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Source URL for this candidate. Used as the unique key.",
            },
            "name": {
                "type": "string",
                "description": "Display name (company, place, person, product, etc.).",
            },
            "why_fits": {
                "type": "string",
                "description": "1-2 sentences grounded in fetched content explaining the fit.",
            },
            "score": {
                "type": "number",
                "description": "Relevance to the stated goal, in [0, 1]. 0 = irrelevant, 1 = perfect fit.",
            },
            "axes": {
                "type": "object",
                "description": "Optional per-dimension breakdown of the score.",
            },
        },
        "required": ["url", "name", "why_fits", "score"],
    },
}


def add_candidate_run(
    url: str,
    name: str,
    why_fits: str,
    score: float,
    axes: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Validate the proposed candidate and echo it back.

    Returns a dict with `ok: True` + the normalized candidate, OR `error`.
    Validation is strict — out-of-range score returns an error rather than
    clamping silently (silent clamping would hide model bugs).
    """
    if not isinstance(url, str) or not url.strip():
        return {"error": "url is required and must be non-empty"}
    if not isinstance(name, str) or not name.strip():
        return {"error": "name is required and must be non-empty"}
    if not isinstance(why_fits, str) or not why_fits.strip():
        return {"error": "why_fits is required and must be non-empty"}
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return {"error": "score must be a number"}
    # NaN passes `< 0 or > 1` silently (any comparison with NaN is False) and
    # poisons downstream sum/sort logic. Reject explicitly. Same for ±inf.
    if not math.isfinite(float(score)):
        return {"error": f"score must be a finite number, got {score}"}
    if score < 0 or score > 1:
        return {"error": f"score must be in [0, 1], got {score}"}

    candidate = {
        "url": url.strip(),
        "name": name.strip(),
        "why_fits": why_fits.strip(),
        "score": float(score),
    }
    if axes is not None:
        if not isinstance(axes, dict):
            return {"error": "axes must be a dict when provided"}
        candidate["axes"] = {k: float(v) for k, v in axes.items() if isinstance(v, (int, float))}

    return {"ok": True, "candidate": candidate}


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

FINALIZE_TOOL_SCHEMA = {
    "name": "finalize",
    "description": (
        "Signal that the goal is met and the candidates list should be "
        "emitted as the final output. Call this AFTER you have added enough "
        "candidates (typically 10 for a 'find 10 X' prompt) at acceptable "
        "scores. The loop exits cleanly and renders the ranked list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def finalize_run() -> dict[str, Any]:
    """No-op signaling tool. Returns a flag the loop's dispatcher reads."""
    return {"finalized": True}


# ---------------------------------------------------------------------------
# Module-level adapters matching the existing tools/<name>.py shape
# (search.run, fetch.run, summarize.run — single `run` symbol per module).
# These are what the loop's TOOL_REGISTRY looks up at dispatch time. Keeping
# both modules in one file because they share lifecycle and are conceptually
# paired (add → finalize); splitting would force test fixture duplication.
# ---------------------------------------------------------------------------

class _AddCandidateModule:
    TOOL_SCHEMA = ADD_CANDIDATE_TOOL_SCHEMA

    @staticmethod
    def run(**kwargs) -> dict[str, Any]:
        return add_candidate_run(**kwargs)


class _FinalizeModule:
    TOOL_SCHEMA = FINALIZE_TOOL_SCHEMA

    @staticmethod
    def run(**kwargs) -> dict[str, Any]:
        return finalize_run()


add_candidate = _AddCandidateModule()
finalize = _FinalizeModule()
