"""Operator simulation + composite verdict for the canary mechanism.

Two public functions:

- `coverage_score(prompt, briefing)`: one Haiku call. Simulates a RapidSOS
  operator reading the briefing in 3 minutes and asking 5 follow-up questions.
  Each YES answer must cite a `[^N]` footnote in the briefing — the
  groundedness check defeats the LLM-judge generosity bias adversarial
  flagged. Coverage = (answered AND grounded) / total.

- `composite_score(eval_record, weights)`: pure function. Combines operator
  coverage + structure score + token efficiency + summarize-error count into
  a single ordered float in [0, 1] for canary verdict.

Together these replace the saturated structure-score-only canary. The
composite formula is intentionally simple — interpretability beats
sophistication when the metric is the load-bearing trust surface.

See docs/plans/2026-05-27-002-feat-meta-eval-tier2-reframings-plan.md U1.
"""

import hashlib
import json
import os
import re

from agent import llm

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Default composite weights. Tuneable; chosen so operator-utility dominates
# without letting it monopolize the verdict when noisy.
DEFAULT_WEIGHTS = {
    "operator": 0.5,
    "structure": 0.2,
    "efficiency": 0.2,
    "errors": 0.1,
}

# Verdict thresholds — anchored to "noticeable change," not "any change."
PROMOTE_DELTA = 0.05
DISCARD_DELTA = -0.10

# Skip LLM call for trivially short briefings (likely fallback stubs)
MIN_BRIEFING_LEN = 100

# Token normalization: divide raw token delta by this to land in [-1, 1] ish.
# 50k = the rough scale of a single full agent run's input tokens.
_TOKEN_NORM = 50_000


_coverage_cache: dict[str, dict] = {}


OPERATOR_SIM_PROMPT = """You are a RapidSOS operator who has 3 minutes to triage <briefing>.

User prompt:
{prompt}

Briefing:
{briefing}

Step 1: List 5 follow-up questions you'd most want answered in your next 30 minutes of work.
Step 2: For each, answer YES/NO: does the briefing as-written already answer it?
Step 3: For each YES, cite the [^N] footnote in the briefing that supports it.
        If you cannot cite a footnote, change the answer to NO — the briefing
        is implying without sourcing, and that does not count as answered.

Return STRICT JSON only (no markdown fence, no commentary outside the JSON):

{{
  "questions": ["q1", "q2", "q3", "q4", "q5"],
  "answered": [true, false, true, true, false],
  "citations": ["[^2]", "", "[^4]", "[^1,3]", ""],
  "coverage": 0.4
}}

Coverage MUST equal (count of answered=true AND citations non-empty) / 5.
"""


def _hash_briefing(briefing: str) -> str:
    """Cache key for coverage_score results."""
    return hashlib.sha256(briefing.encode("utf-8")).hexdigest()[:16]


def coverage_score(prompt: str, briefing: str, *, use_llm: bool = True) -> dict:
    """Run operator simulation. Returns:

      {
        "coverage": float in [0, 1],
        "questions": [...],
        "answered": [...],
        "citations": [...],
        "cached": bool,
        "error": str | None,
      }

    Truncates briefings >8000 chars to keep prompt sizes bounded.
    """
    if not briefing or len(briefing) < MIN_BRIEFING_LEN:
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "skip_reason": "briefing too short",
        }

    cache_key = _hash_briefing(briefing)
    if cache_key in _coverage_cache:
        cached = dict(_coverage_cache[cache_key])
        cached["cached"] = True
        return cached

    if not use_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "skip_reason": "no_llm_mode",
        }

    user_message = OPERATOR_SIM_PROMPT.format(
        prompt=prompt[:500],
        briefing=briefing[:8000],
    )

    try:
        response = llm.call_with_retry({
            "model": HAIKU_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_message}],
        })
    except Exception as exc:
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "error": f"non-JSON response: {exc}",
        }

    # Defensive normalization: recompute coverage from answered+citations
    # to defeat the case where the LLM lies about its own coverage field.
    answered = parsed.get("answered") or []
    citations = parsed.get("citations") or []
    n = max(len(answered), len(citations), 5)
    grounded = sum(
        1
        for i in range(min(len(answered), len(citations)))
        if answered[i] and (citations[i] or "").strip()
    )
    coverage = grounded / n if n else 0.0

    result = {
        "coverage": round(coverage, 4),
        "questions": parsed.get("questions") or [],
        "answered": answered,
        "citations": citations,
        "reported_coverage": parsed.get("coverage"),
        "cached": False,
    }
    _coverage_cache[cache_key] = result
    return result


def composite_score(eval_record: dict, weights: dict | None = None) -> dict:
    """Pure function. Combine signals into a single ordered float in [0, 1].

    Expected eval_record shape (defensive — missing fields default to neutral):
      {
        "operator_sim": {"coverage": float},        # from coverage_score
        "structure": {"score": int 0..5},           # from evaluate.check_structure
        "tokens": {"input": int, "output": int},    # from loop run
        "trace_issues": [{"tag": "..."}],           # for error count
      }

    Returns {composite, breakdown, weights_used}.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    breakdown = {}

    op = (eval_record.get("operator_sim") or {}).get("coverage")
    breakdown["operator"] = float(op) if op is not None else 0.5  # neutral default

    struct = (eval_record.get("structure") or {}).get("score")
    breakdown["structure"] = (struct / 5.0) if isinstance(struct, (int, float)) else 0.5

    # Efficiency: 0 tokens = perfect (1.0); _TOKEN_NORM tokens = neutral (0.5);
    # >2*_TOKEN_NORM = floored at 0.0.
    tokens = eval_record.get("tokens") or {}
    total_tokens = (tokens.get("input") or 0) + (tokens.get("output") or 0)
    if total_tokens <= 0:
        breakdown["efficiency"] = 0.5
    else:
        breakdown["efficiency"] = max(0.0, min(1.0, 1.0 - total_tokens / (2 * _TOKEN_NORM)))

    # Errors: 0 errors = perfect; 1 = penalized; ≥3 = floored
    n_errors = len(eval_record.get("trace_issues") or [])
    breakdown["errors"] = max(0.0, 1.0 - n_errors / 3.0)

    composite = (
        w["operator"] * breakdown["operator"]
        + w["structure"] * breakdown["structure"]
        + w["efficiency"] * breakdown["efficiency"]
        + w["errors"] * breakdown["errors"]
    )

    return {
        "composite": round(composite, 4),
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "weights_used": w,
    }


def classify_composite(baseline_composite: float, candidate_composite: float) -> tuple[str, str]:
    """Deterministic verdict from composite deltas.

    Returns (decision, reason). Delta is rounded to 4 decimals before
    comparison so float-arithmetic edge cases (0.55 - 0.5 == 0.05000…04)
    don't push borderline cases across the threshold.
    """
    delta = round(candidate_composite - baseline_composite, 4)
    if delta > PROMOTE_DELTA:
        return "promote", f"composite improved by {delta:+.3f} (threshold {PROMOTE_DELTA})"
    if delta < DISCARD_DELTA:
        return "discard", f"composite regressed by {delta:+.3f} (threshold {DISCARD_DELTA})"
    return "gate", f"composite delta {delta:+.3f} within noise band ({DISCARD_DELTA}, {PROMOTE_DELTA}]"


def reset_cache() -> None:
    """Test helper: clear the coverage cache between scenarios."""
    _coverage_cache.clear()
