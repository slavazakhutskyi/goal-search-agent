"""Tests for the operator simulation + composite verdict.

LLM call is mocked. Composite logic is pure-function tested.
"""

import json
import types

import pytest

from agent import operator_sim


# ---------- coverage_score ----------

GOOD_BRIEFING = (
    "# Briefing — Test\n\n"
    "## TL;DR\nCarbyne raised funding [^1] and partnered with Axon [^2].\n\n"
    "## Key themes\n- Funding round [^1]\n- Partnership [^2]\n\n"
    "## Sentiment\n**Positive** — strong coverage.\n\n"
    "## Sources\n[^1]: [Funding](u1) — d, 2026-01-01\n[^2]: [Axon deal](u2) — d, 2026-01-02\n"
)


def test_coverage_skips_short_briefing(temp_data_dir):
    """Briefings < 100 chars are likely fallback stubs → coverage 0 without LLM call."""
    operator_sim.reset_cache()
    out = operator_sim.coverage_score("prompt", "short")
    assert out["coverage"] == 0.0
    assert out["skip_reason"] == "briefing too short"


def test_coverage_skips_no_llm_mode(temp_data_dir, monkeypatch):
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    out = operator_sim.coverage_score("prompt", GOOD_BRIEFING, use_llm=True)
    assert out["coverage"] == 0.0
    assert out["skip_reason"] == "no_llm_mode"


def _stub_response(payload: dict):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=json.dumps(payload))]
    )


def test_coverage_computes_grounded_ratio(mocker, temp_data_dir, monkeypatch):
    """3 of 5 answered=YES AND have citations → coverage = 0.6."""
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "questions": ["q1", "q2", "q3", "q4", "q5"],
            "answered": [True, True, True, False, False],
            "citations": ["[^1]", "[^2]", "[^3]", "", ""],
            "coverage": 0.6,  # LLM-reported
        }),
    )
    out = operator_sim.coverage_score("p", GOOD_BRIEFING)
    assert out["coverage"] == 0.6
    assert out["reported_coverage"] == 0.6


def test_coverage_groundedness_overrides_yes_without_citation(mocker, temp_data_dir, monkeypatch):
    """YES with empty citation does NOT count toward coverage (defeats nonsense-judge bias)."""
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "questions": ["q1", "q2", "q3", "q4", "q5"],
            "answered": [True, True, True, True, True],  # judge says all answered
            "citations": ["[^1]", "", "", "", ""],  # but only 1 cites a footnote
            "coverage": 1.0,  # judge tries to claim 1.0
        }),
    )
    out = operator_sim.coverage_score("p", GOOD_BRIEFING)
    # Grounded coverage = 1 / 5 = 0.2, NOT the 1.0 the judge claimed
    assert out["coverage"] == 0.2


def test_coverage_caches_by_prompt_and_briefing(mocker, temp_data_dir, monkeypatch):
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    spy = mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "questions": ["q"] * 5,
            "answered": [True] * 5,
            "citations": ["[^1]"] * 5,
            "coverage": 1.0,
        }),
    )
    operator_sim.coverage_score("p", GOOD_BRIEFING)
    operator_sim.coverage_score("p", GOOD_BRIEFING)  # same prompt + briefing → cache hit
    assert spy.call_count == 1


def test_coverage_cache_keys_on_prompt_too(mocker, temp_data_dir, monkeypatch):
    """Different prompts on the SAME briefing must miss the cache.

    The operator's follow-up questions depend on the prompt, not just the
    briefing. Hashing only the briefing returns stale answers for
    compare()/multi-prompt canary workflows.
    """
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    spy = mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "questions": ["q"] * 5,
            "answered": [True] * 5,
            "citations": ["[^1]"] * 5,
            "coverage": 1.0,
        }),
    )
    operator_sim.coverage_score("prompt A", GOOD_BRIEFING)
    operator_sim.coverage_score("prompt B", GOOD_BRIEFING)  # different prompt → miss
    assert spy.call_count == 2


def test_coverage_denominator_fixed_at_5_when_llm_returns_fewer(mocker, temp_data_dir, monkeypatch):
    """LLM returns 4 items → coverage uses /5 anyway. Defeats verbosity drift."""
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "questions": ["q1", "q2", "q3", "q4"],     # only 4 questions
            "answered": [True, True, True, True],       # all answered
            "citations": ["[^1]", "[^2]", "[^3]", "[^4]"],  # all cited
            "coverage": 1.0,                            # LLM claims 4/4
        }),
    )
    out = operator_sim.coverage_score("p", GOOD_BRIEFING)
    # Grounded = 4, denominator FIXED at EXPECTED_QUESTIONS=5 → 0.8 not 1.0
    assert out["coverage"] == 0.8


def test_coverage_denominator_fixed_at_5_when_llm_returns_more(mocker, temp_data_dir, monkeypatch):
    """LLM returns 7 items → only first 5 count. Defeats verbosity drift."""
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "questions": ["q"] * 7,
            "answered": [True] * 7,
            "citations": ["[^1]"] * 7,
            "coverage": 1.0,
        }),
    )
    out = operator_sim.coverage_score("p", GOOD_BRIEFING)
    # Capped at 5: 5 grounded / 5 = 1.0
    assert out["coverage"] == 1.0


def test_coverage_handles_malformed_json(mocker, temp_data_dir, monkeypatch):
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="not json")]
        ),
    )
    out = operator_sim.coverage_score("p", GOOD_BRIEFING)
    assert out["coverage"] == 0.0
    assert "non-JSON" in out["error"]


# ---------- composite_score ----------

def test_composite_neutral_defaults_for_missing_fields():
    """Missing operator_sim, structure, tokens, trace_issues → all 0.5 (neutral)."""
    out = operator_sim.composite_score({})
    # All four components default to 0.5 (or 1.0 for errors with 0 issues)
    b = out["breakdown"]
    assert b["operator"] == 0.5
    assert b["structure"] == 0.5
    assert b["efficiency"] == 0.5
    assert b["errors"] == 1.0  # 0 trace_issues = perfect errors score
    # Composite: 0.5*0.5 + 0.2*0.5 + 0.2*0.5 + 0.1*1.0 = 0.55
    assert out["composite"] == 0.55


def test_composite_high_operator_dominates():
    """coverage=1.0 with neutral other signals → composite well above 0.5."""
    out = operator_sim.composite_score({
        "operator_sim": {"coverage": 1.0},
        "structure": {"score": 5},
    })
    # 0.5*1.0 + 0.2*1.0 + 0.2*0.5 + 0.1*1.0 = 0.9
    assert out["composite"] == 0.9


def test_composite_token_efficiency_punishes_bloat():
    """High token usage drops the efficiency component toward 0."""
    out = operator_sim.composite_score({
        "operator_sim": {"coverage": 0.5},
        "structure": {"score": 3},
        "tokens": {"input": 100_000, "output": 50_000},  # 150k > 2*_TOKEN_NORM
    })
    assert out["breakdown"]["efficiency"] == 0.0


def test_composite_errors_penalize_summarize_failures():
    """3+ trace_issues → errors component floored at 0."""
    out = operator_sim.composite_score({
        "trace_issues": [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}, {"tag": "d"}],
    })
    assert out["breakdown"]["errors"] == 0.0


def test_composite_weights_override():
    """Custom weights change verdict ordering on borderline cases."""
    record = {"operator_sim": {"coverage": 1.0}, "structure": {"score": 0}}
    default = operator_sim.composite_score(record)
    op_heavy = operator_sim.composite_score(record, weights={"operator": 1.0, "structure": 0, "efficiency": 0, "errors": 0})
    assert op_heavy["composite"] > default["composite"]


# classify_composite tests deleted with the function (M-02/AC-6). Verdict
# logic is exercised through agent.meta_eval.classify_canary in test_meta_eval.


# ---------- U4: output-shape detection ----------

CANDIDATES_MARKDOWN = (
    "# Goal — find 10 things\n"
    "*Generated x · 10 candidates · sorted by score desc*\n\n"
    "## Candidates\n\n"
    "1. **Acme** (https://a.example) — score 0.90\n"
    "   matches X requirement\n"
)

BRIEFING_MARKDOWN = (
    "# Briefing — competitive intel\n\n"
    "## TL;DR\nThings.\n\n"
    "## Key themes\n- A [^1]\n\n"
    "## Sentiment\n**Neutral** — ok.\n\n"
    "## Sources\n[^1]: [A](https://a.example)\n"
)


def test_detect_output_shape_candidates():
    assert operator_sim.detect_output_shape(CANDIDATES_MARKDOWN) == "candidates"


def test_detect_output_shape_briefing():
    assert operator_sim.detect_output_shape(BRIEFING_MARKDOWN) == "briefing"


def test_detect_output_shape_unknown_when_empty():
    assert operator_sim.detect_output_shape("") == "unknown"
    assert operator_sim.detect_output_shape(None) == "unknown"
    assert operator_sim.detect_output_shape("plain text") == "unknown"


def test_detect_output_shape_hybrid_prefers_briefing():
    """If both markers present (unusual), prefer briefing (more structured)."""
    hybrid = CANDIDATES_MARKDOWN + "\n" + BRIEFING_MARKDOWN
    assert operator_sim.detect_output_shape(hybrid) == "briefing"


# ---------- U4: goal_coverage_score ----------

def test_goal_coverage_skips_short_text(temp_data_dir):
    operator_sim.reset_cache()
    out = operator_sim.goal_coverage_score("find 10 things", "short")
    assert out["coverage"] == 0.0
    assert out["skip_reason"] == "candidates text too short"


def test_goal_coverage_no_llm_mode(temp_data_dir, monkeypatch):
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    out = operator_sim.goal_coverage_score("find 10 things", CANDIDATES_MARKDOWN)
    assert out["coverage"] == 0.0
    assert out["skip_reason"] == "no_llm_mode"


def test_goal_coverage_groundedness_gate(mocker, temp_data_dir, monkeypatch):
    """YES without citation → not counted."""
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    mocker.patch(
        "agent.operator_sim.llm.call_with_retry",
        return_value=_stub_response({
            "checks": ["c1", "c2", "c3", "c4", "c5"],
            "answered": [True, True, True, True, True],
            "citations": ["Acme", "", "", "", ""],  # only 1 grounded
            "coverage": 1.0,
        }),
    )
    out = operator_sim.goal_coverage_score("find 10 things", CANDIDATES_MARKDOWN)
    assert out["coverage"] == 0.2  # grounded 1/5, not LLM-claimed 1.0


def test_goal_coverage_distinguishes_from_coverage_score_via_cache(
    mocker, temp_data_dir, monkeypatch
):
    """Same prompt+text should produce different cache keys for the two scorers."""
    operator_sim.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    response = _stub_response({
        "questions": ["q"] * 5,
        "checks": ["c"] * 5,
        "answered": [True] * 5,
        "citations": ["x"] * 5,
        "coverage": 1.0,
    })
    spy = mocker.patch("agent.operator_sim.llm.call_with_retry", return_value=response)
    # Both calls use same prompt + text but different scorers → both should
    # hit LLM (different cache keys due to GOAL: prefix).
    operator_sim.coverage_score("p", CANDIDATES_MARKDOWN)
    operator_sim.goal_coverage_score("p", CANDIDATES_MARKDOWN)
    assert spy.call_count == 2
