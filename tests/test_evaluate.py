"""Tests for the self-evaluation module. Pure-function layers only — the LLM
judge is exercised via a mock to verify JSON parsing, not Anthropic itself."""

import json

import pytest

from agent import evaluate


# ---------- Layer 1: structure check ----------

GOOD_BRIEFING = """# Briefing — Test prompt

## TL;DR
This is the gist in one short paragraph. It says useful things.

## Key themes
- Theme one with citation [^1]
- Theme two with citation [^2]
- Theme three with citation [^3]

## Notable mentions
> A quote — Source[^1]

## Sentiment
**Positive** — coverage skews favorable based on most sources.

## Sources
[^1]: [Source A](https://a.example) — a.example, 2026-01-01
[^2]: [Source B](https://b.example) — b.example, 2026-01-02
[^3]: [Source C](https://c.example) — c.example, 2026-01-03
"""

BROKEN_BRIEFING = """# Briefing — Test prompt

Random text without the expected structure.
"""


def test_structure_check_passes_well_formed_briefing():
    result = evaluate.check_structure(GOOD_BRIEFING)
    assert result["score"] == 5
    assert result["fails"] == []
    assert "tldr_section_present" in result["passes"]
    assert "sentiment_labeled" in result["passes"]


def test_structure_check_flags_missing_sections():
    result = evaluate.check_structure(BROKEN_BRIEFING)
    assert result["score"] == 0
    assert "tldr_section_missing" in result["fails"]
    assert "sentiment_missing_or_unlabeled" in result["fails"]
    assert "sources_section_missing" in result["fails"]


def test_structure_check_flags_uncited_themes():
    briefing = GOOD_BRIEFING.replace("Theme two with citation [^2]", "Theme two no citation")
    result = evaluate.check_structure(briefing)
    assert any("uncited_themes" in f for f in result["fails"])


# ---------- Layer 2: trace critique ----------

def test_trace_critique_flags_missing_documents_arg():
    run_log = {
        "tool_calls": [
            {"name": "summarize", "error": "bad tool input for summarize: run() missing 1 required positional argument: 'documents'"},
            {"name": "summarize", "error": "bad tool input for summarize: run() missing 1 required positional argument: 'documents'"},
        ],
    }
    issues = evaluate.critique_trace(run_log)
    assert any(i["tag"] == "summarize_missing_documents" for i in issues)


def test_trace_critique_flags_search_budget():
    run_log = {
        "tool_calls": [{"name": "search", "error": None} for _ in range(8)],
    }
    issues = evaluate.critique_trace(run_log)
    assert any(i["tag"] == "search_budget_exceeded" for i in issues)


def test_trace_critique_flags_403():
    run_log = {
        "tool_calls": [
            {"name": "fetch", "error": "fetch failed: HTTPStatusError(\"403 Forbidden\")"},
        ],
    }
    issues = evaluate.critique_trace(run_log)
    assert any(i["tag"] == "fetch_403_dropped" for i in issues)


def test_trace_critique_flags_cap_without_summarize():
    run_log = {
        "iterations": 12,
        "tool_calls": [
            {"name": "search", "error": None},
            {"name": "fetch", "error": None},
        ],
    }
    issues = evaluate.critique_trace(run_log)
    assert any(i["tag"] == "cap_hit_without_summarize" for i in issues)


def test_trace_critique_clean_run_has_no_issues():
    run_log = {
        "iterations": 4,
        "status": "complete",
        "tool_calls": [
            {"name": "search", "error": None},
            {"name": "fetch", "error": None},
            {"name": "fetch", "error": None},
            {"name": "summarize", "error": None},
        ],
    }
    assert evaluate.critique_trace(run_log) == []


# ---------- Layer 3: LLM judge (parses Haiku output) ----------

class _StubBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _StubResponse:
    def __init__(self, text):
        self.content = [_StubBlock(text)]


def test_llm_judge_parses_clean_json(mocker):
    payload = {"usefulness": 4, "missing": ["a", "b"], "next_run_suggestion": "Fetch more recent sources."}
    mocker.patch(
        "agent.llm.call_with_retry",
        return_value=_StubResponse(json.dumps(payload)),
    )
    out = evaluate.llm_judge("prompt", "briefing")
    assert out["usefulness"] == 4
    assert out["next_run_suggestion"] == "Fetch more recent sources."


def test_llm_judge_strips_markdown_fences(mocker):
    payload = {"usefulness": 3, "missing": [], "next_run_suggestion": "Cite every theme."}
    fenced = f"```json\n{json.dumps(payload)}\n```"
    mocker.patch(
        "agent.llm.call_with_retry",
        return_value=_StubResponse(fenced),
    )
    out = evaluate.llm_judge("prompt", "briefing")
    assert out["usefulness"] == 3


def test_llm_judge_surfaces_parse_failure(mocker):
    mocker.patch(
        "agent.llm.call_with_retry",
        return_value=_StubResponse("not json at all"),
    )
    out = evaluate.llm_judge("prompt", "briefing")
    assert "error" in out


# ---------- Compounding loop: lessons persistence ----------

def test_lessons_round_trip(temp_data_dir, mocker):
    # Skip the LLM judge for determinism.
    mocker.patch.dict("os.environ", {}, clear=False)
    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False)
    run_log = {
        "prompt": "X",
        "briefing_path": "x.md",
        "iterations": 12,
        "tool_calls": [
            {"name": "summarize", "error": "bad tool input for summarize: run() missing 1 required positional argument: 'documents'"},
        ],
    }
    eval_record = evaluate.run_self_eval(run_log, BROKEN_BRIEFING, use_llm_judge=False)
    evaluate.save_eval(eval_record)

    lessons = evaluate.load_recent_lessons()
    assert any("summarize" in lesson.lower() for lesson in lessons)


def test_lessons_block_empty_returns_empty_string():
    assert evaluate.lessons_block([]) == ""


def test_lessons_block_renders_bullets():
    block = evaluate.lessons_block(["Do X.", "Do Y."])
    assert "## Recent lessons" in block
    assert "- Do X." in block
    assert "- Do Y." in block


def test_load_recent_lessons_deduplicates(temp_data_dir):
    # Two evals with identical suggestion → one lesson
    record = {
        "trace_issues": [
            {"tag": "t", "detail": "d", "suggestion": "Always call summarize with documents."},
        ],
        "judge": {"skipped": True},
    }
    evaluate.save_eval(record)
    evaluate.save_eval(record)
    lessons = evaluate.load_recent_lessons()
    assert lessons.count("Always call summarize with documents.") == 1
