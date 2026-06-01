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
TOKEN_NORM = 50_000

# Cache size cap. Prevents unbounded growth in long-running processes; FIFO
# eviction is sufficient for the typical canary workload (a handful of
# distinct briefings per session).
MAX_CACHE_SIZE = 256

# Expected number of operator follow-up questions. Coverage denominator is
# FIXED at this — never derived from LLM response length — so LLM verbosity
# (returning 4 or 7 items) cannot drift the metric off-contract.
EXPECTED_QUESTIONS = 5


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


def _cache_key(prompt: str, briefing: str) -> str:
    """Cache key includes BOTH prompt and briefing. The operator's follow-up
    questions are derived from the prompt; two different prompts on the same
    briefing must produce different question sets and therefore different
    coverage scores. Hashing only the briefing (the previous bug) returned
    stale answers for any compare()/multi-prompt canary workflow.
    """
    combined = f"{prompt}\x00{briefing}".encode("utf-8")
    return hashlib.sha256(combined).hexdigest()[:16]


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

    cache_key = _cache_key(prompt, briefing)
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

    # Narrow exception handling — match the convention in agent/tools/search.py
    # and agent/tools/fetch.py: graceful error dict, never raise. Keep a final
    # broad except as a safety net but typed errors get prioritized handling.
    try:
        response = llm.call_with_retry({
            "model": HAIKU_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_message}],
        })
    except (RuntimeError, ValueError) as exc:
        # RuntimeError covers the typed "ANTHROPIC_API_KEY not set" from llm.client.
        # ValueError covers bad-input cases (malformed kwargs).
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:  # pragma: no cover — anthropic transient/network errors
        # llm.call_with_retry already retries rate-limit/timeout/connection. If we
        # land here, retries were exhausted. Surface as error dict so the canary
        # can record the failure without crashing the autonomous loop.
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
    # Denominator is FIXED at EXPECTED_QUESTIONS (5) — not derived from the
    # response — so LLM verbosity (returning 4 or 7 items) cannot drift the
    # metric off-contract. Extra items past 5 are ignored; missing items count
    # as unanswered.
    answered = parsed.get("answered") or []
    citations = parsed.get("citations") or []
    grounded = sum(
        1
        for i in range(min(len(answered), len(citations), EXPECTED_QUESTIONS))
        if answered[i] and (citations[i] or "").strip()
    )
    coverage = grounded / EXPECTED_QUESTIONS

    # Capture Haiku token usage so callers (run_canary, autonomous cost-cap)
    # can include operator_sim's API cost in their accounting. Without this,
    # autonomous() silently undercounts by ~$0.01 per canary call.
    usage = getattr(response, "usage", None)
    tokens = None
    if usage is not None:
        tokens = {
            "input": getattr(usage, "input_tokens", 0),
            "output": getattr(usage, "output_tokens", 0),
        }

    result = {
        "coverage": round(coverage, 4),
        "questions": parsed.get("questions") or [],
        "answered": answered,
        "citations": citations,
        "reported_coverage": parsed.get("coverage"),
        "tokens": tokens,
        "cached": False,
    }
    # FIFO eviction when over the cap. Simple and bounded.
    if len(_coverage_cache) >= MAX_CACHE_SIZE:
        # Drop oldest entry. Dicts preserve insertion order in Python 3.7+.
        try:
            oldest = next(iter(_coverage_cache))
            del _coverage_cache[oldest]
        except StopIteration:
            pass
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
        breakdown["efficiency"] = max(0.0, min(1.0, 1.0 - total_tokens / (2 * TOKEN_NORM)))

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


# classify_composite was a parallel implementation of meta_eval.classify_canary.
# Deleted per code review M-02/AC-6 — production canary always calls
# meta_eval.classify_canary (which handles type=delete override, empty-K guard,
# and K-averaging). A standalone two-arg classifier here would silently drift
# from the canary's actual decision logic. Use classify_canary directly.


def reset_cache() -> None:
    """Test helper: clear the coverage cache between scenarios."""
    _coverage_cache.clear()


# ============================================================
# U4 — Goal-coverage metric (for goal-search output shape)
# ============================================================

def detect_output_shape(text: str) -> str:
    """Identify the output shape of a briefing/candidates_list markdown.

    Returns one of "candidates", "briefing", "unknown". Used by the canary
    verdict dispatch in meta_eval.run_canary to pick the right scorer.
    Heuristic — checks for distinctive section headers rather than parsing.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    has_candidates_header = "## Candidates" in text
    has_briefing_markers = "## TL;DR" in text or "## Sources" in text
    if has_candidates_header and not has_briefing_markers:
        return "candidates"
    if has_briefing_markers and not has_candidates_header:
        return "briefing"
    if has_candidates_header and has_briefing_markers:
        # Both present — unusual but possible if a model emits a hybrid.
        # Prefer briefing (more structured shape).
        return "briefing"
    return "unknown"


GOAL_COVERAGE_PROMPT = """You are evaluating a goal-completion search agent's output.

User prompt (the goal):
{prompt}

Candidates list the agent produced:
{candidates_text}

Step 1: List 5 yes/no checks for whether this list satisfies the goal.
        Cover: count vs requested N, diversity of candidates, quality of
        why_fits explanations, score distribution, alignment with criteria.
Step 2: For each check, answer YES/NO. Be strict — answer YES only when
        the list clearly passes the check.
Step 3: For each YES, cite a specific candidate name or score from the list
        as evidence. If you cannot cite, change the answer to NO.

Return STRICT JSON only (no markdown fence):

{{
  "checks": ["...", "...", "...", "...", "..."],
  "answered": [true, false, true, true, false],
  "citations": ["Acme score 0.9", "", "WidgetCo", "names diverse", ""],
  "coverage": 0.6
}}

Coverage MUST equal (count of answered=true AND citations non-empty) / 5.
"""


def goal_coverage_score(prompt: str, candidates_text: str, *, use_llm: bool = True) -> dict:
    """Goal-coverage analog of `coverage_score` for the candidates_list shape.

    Returns the same dict shape as `coverage_score`: {coverage, checks (or
    questions), answered, citations, cached, error/skip_reason}. Defensive
    recomputation against EXPECTED_QUESTIONS (5) and groundedness gate
    (YES needs non-empty citation) — same defenses as coverage_score.

    Distinct from coverage_score because the prompt + axes are different:
    candidates have count/diversity/quality dimensions, briefings have
    operator-utility dimensions.
    """
    if not candidates_text or len(candidates_text) < MIN_BRIEFING_LEN:
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "skip_reason": "candidates text too short",
        }

    cache_key = _cache_key(f"GOAL:{prompt}", candidates_text)
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

    user_message = GOAL_COVERAGE_PROMPT.format(
        prompt=prompt[:500],
        candidates_text=candidates_text[:8000],
    )

    try:
        response = llm.call_with_retry({
            "model": HAIKU_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_message}],
        })
    except (RuntimeError, ValueError) as exc:
        return {
            "coverage": 0.0,
            "questions": [],
            "answered": [],
            "citations": [],
            "cached": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:  # pragma: no cover
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

    # Same groundedness gate as coverage_score: YES counts only when citation
    # is non-empty AND denominator is FIXED at EXPECTED_QUESTIONS.
    answered = parsed.get("answered") or []
    citations = parsed.get("citations") or []
    grounded = sum(
        1
        for i in range(min(len(answered), len(citations), EXPECTED_QUESTIONS))
        if answered[i] and (citations[i] or "").strip()
    )
    coverage = grounded / EXPECTED_QUESTIONS

    usage = getattr(response, "usage", None)
    tokens = None
    if usage is not None:
        tokens = {
            "input": getattr(usage, "input_tokens", 0),
            "output": getattr(usage, "output_tokens", 0),
        }

    result = {
        "coverage": round(coverage, 4),
        "questions": parsed.get("checks") or parsed.get("questions") or [],
        "answered": answered,
        "citations": citations,
        "reported_coverage": parsed.get("coverage"),
        "tokens": tokens,
        "cached": False,
    }
    if len(_coverage_cache) >= MAX_CACHE_SIZE:
        try:
            oldest = next(iter(_coverage_cache))
            del _coverage_cache[oldest]
        except StopIteration:
            pass
    _coverage_cache[cache_key] = result
    return result
