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


# ---------- U3: few-shot demonstrations with metric_version gate ----------

def _write_edit_history(temp_data_dir, records: list[dict]):
    """Helper: write records to data/logs/edit_history.jsonl."""
    from agent import storage
    path = storage.LOGS / "edit_history.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_runs(temp_data_dir, runs: list[dict]):
    from agent import storage
    path = storage.LOGS / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in runs) + "\n")


def test_load_few_shot_examples_empty_when_no_edit_history(temp_data_dir):
    out = evaluate.load_few_shot_examples()
    assert out["ready"] is False
    assert out["good"] == []
    assert out["bad"] == []


def test_load_few_shot_examples_gates_on_metric_version(temp_data_dir):
    """Records without metric_version=composite-v1 are EXCLUDED entirely.
    Poisoned-label gate per adversarial F3."""
    _write_edit_history(temp_data_dir, [
        {"timestamp": "2026-05-26T10:00:00+00:00", "kept": True,
         "metric_version": "structure-v0"},  # pre-composite — excluded
        {"timestamp": "2026-05-26T11:00:00+00:00", "kept": False},  # no version — excluded
        {"timestamp": "2026-05-26T12:00:00+00:00", "kept": True,
         "metric_version": "composite-v1"},  # included (only 1)
    ])
    out = evaluate.load_few_shot_examples()
    # Only 1 composite-v1 record < min_records=3 → not ready
    assert out["ready"] is False
    assert out["n_composite_records"] == 1


def test_load_few_shot_examples_ready_at_min_records(temp_data_dir):
    """≥3 composite-v1 records → demonstrations block populated."""
    _write_runs(temp_data_dir, [
        {"timestamp": "2026-05-26T09:00:00+00:00", "prompt": "good prompt", "tool_calls": [
            {"name": "search"}, {"name": "fetch"}, {"name": "summarize"}
        ], "briefing_path": "data/briefings/g.md"},
        {"timestamp": "2026-05-26T10:30:00+00:00", "prompt": "bad prompt", "tool_calls": [
            {"name": "search"}, {"name": "search"}, {"name": "search"}, {"name": "summarize", "error": "x"}
        ], "briefing_path": "data/briefings/b.md"},
        {"timestamp": "2026-05-26T11:30:00+00:00", "prompt": "third prompt", "tool_calls": [
            {"name": "summarize"}
        ]},
    ])
    _write_edit_history(temp_data_dir, [
        {"timestamp": "2026-05-26T09:30:00+00:00", "kept": True,
         "metric_version": "composite-v1", "composite_score": 0.78},
        {"timestamp": "2026-05-26T11:00:00+00:00", "kept": False,
         "metric_version": "composite-v1", "composite_score": 0.30},
        {"timestamp": "2026-05-26T12:00:00+00:00", "kept": True,
         "metric_version": "composite-v1", "composite_score": 0.82},
    ])
    out = evaluate.load_few_shot_examples()
    assert out["ready"] is True
    assert out["n_composite_records"] == 3
    # Two kept records → 2 good (default n_good=2)
    assert len(out["good"]) == 2
    # One reverted record → 1 bad
    assert len(out["bad"]) == 1


def test_load_few_shot_examples_skips_when_no_matching_run(temp_data_dir):
    """edit_history record with no run within the 24h window → skipped silently."""
    _write_runs(temp_data_dir, [
        # All runs are >24h before the edit timestamps below
        {"timestamp": "2026-01-01T00:00:00+00:00", "prompt": "ancient", "tool_calls": []},
    ])
    _write_edit_history(temp_data_dir, [
        {"timestamp": "2026-05-26T10:00:00+00:00", "kept": True,
         "metric_version": "composite-v1"},
        {"timestamp": "2026-05-26T11:00:00+00:00", "kept": True,
         "metric_version": "composite-v1"},
        {"timestamp": "2026-05-26T12:00:00+00:00", "kept": True,
         "metric_version": "composite-v1"},
    ])
    out = evaluate.load_few_shot_examples()
    # Ready (3 composite records exist) but no examples built (no matching runs)
    assert out["ready"] is True
    assert out["good"] == []
    assert out["bad"] == []


def test_demonstrations_block_empty_when_not_ready(temp_data_dir):
    """Until min_records hit, demonstrations_block returns empty string."""
    assert evaluate.demonstrations_block() == ""


def test_demonstrations_block_renders_good_and_bad(temp_data_dir):
    """Once ready, block contains both good runs and anti-patterns sections."""
    examples = {
        "ready": True,
        "n_composite_records": 3,
        "good": [{
            "edit_timestamp": "2026-05-26T09:30:00+00:00",
            "composite_score": 0.78,
            "prompt": "good prompt", "trace": "1 search · 2 fetch · 1 summarize",
            "briefing_snippet": "TL;DR: results...",
        }],
        "bad": [{
            "edit_timestamp": "2026-05-26T11:00:00+00:00",
            "composite_score": 0.30,
            "prompt": "bad prompt", "trace": "3 search · 1 error(s)",
            "briefing_snippet": "TL;DR: thin...",
        }],
    }
    block = evaluate.demonstrations_block(examples)
    assert "Good runs (kept after canary)" in block
    assert "Anti-patterns" in block
    assert "good prompt" in block
    assert "bad prompt" in block
    assert "1 search · 2 fetch · 1 summarize" in block


def test_load_few_shot_examples_sorts_by_composite_score(temp_data_dir):
    """M-09 regression: good examples sorted by composite_score DESC, bad ASC.

    Previously the slice took whatever order was on disk, so few-shot examples
    were the most-recent kept/reverted edits — NOT the highest-scoring exemplars
    or worst regressions the docstring claims.
    """
    _write_runs(temp_data_dir, [
        {"timestamp": "2026-05-26T09:00:00+00:00", "prompt": "low good", "tool_calls": [],
         "briefing_path": "data/briefings/x1.md"},
        {"timestamp": "2026-05-26T10:00:00+00:00", "prompt": "high good", "tool_calls": [],
         "briefing_path": "data/briefings/x2.md"},
        {"timestamp": "2026-05-26T11:00:00+00:00", "prompt": "mild bad", "tool_calls": [],
         "briefing_path": "data/briefings/x3.md"},
        {"timestamp": "2026-05-26T12:00:00+00:00", "prompt": "severe bad", "tool_calls": [],
         "briefing_path": "data/briefings/x4.md"},
    ])
    # 4 composite-v1 records. Good in mixed order (low first, then high) and
    # bad in mixed order (mild first, then severe). After sort, top good should
    # be "high good" and top bad should be "severe bad".
    _write_edit_history(temp_data_dir, [
        {"timestamp": "2026-05-26T09:30:00+00:00", "kept": True,
         "metric_version": "composite-v1", "composite_score": 0.55},
        {"timestamp": "2026-05-26T10:30:00+00:00", "kept": True,
         "metric_version": "composite-v1", "composite_score": 0.85},
        {"timestamp": "2026-05-26T11:30:00+00:00", "kept": False,
         "metric_version": "composite-v1", "composite_score": 0.40},
        {"timestamp": "2026-05-26T12:30:00+00:00", "kept": False,
         "metric_version": "composite-v1", "composite_score": 0.10},
    ])
    out = evaluate.load_few_shot_examples(n_good=1, n_bad=1)
    assert out["ready"] is True
    # Top good should be the 0.85 scorer ("high good"), not the chronologically-first
    assert out["good"][0]["prompt"] == "high good"
    # Top bad should be the 0.10 scorer ("severe bad"), worst regression first
    assert out["bad"][0]["prompt"] == "severe bad"


def test_evaluate_trace_critique_suggestion_does_not_recommend_old_contract(temp_data_dir):
    """AC-3 regression: the summarize_missing_documents suggestion must NOT
    recommend re-enforcing the old required-documents contract, since U0 made
    documents optional. Otherwise meta-eval can propose SYSTEM_PROMPT edits
    that undo U0 — a self-defeating feedback loop."""
    run_log = {
        "tool_calls": [
            {"name": "summarize", "error": "bad tool input for summarize: run() missing 1 required positional argument: 'documents'"},
        ],
    }
    issues = evaluate.critique_trace(run_log)
    smd = next((i for i in issues if i["tag"] == "summarize_missing_documents"), None)
    assert smd is not None
    # The suggestion text must NOT instruct the model to "pass documents"
    suggestion = smd["suggestion"].lower()
    assert "both prompt and documents" not in suggestion
    # And it SHOULD reference the auto-attach hook as the actual fix surface
    assert "auto-attach" in suggestion or "loop" in suggestion
