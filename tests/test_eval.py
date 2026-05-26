"""Tests for the eval harness — deterministic checks + LLM judge (mocked)."""

from pathlib import Path

from eval import checks
from tests.conftest import make_response, make_text_block


VALID_BRIEFING = """# Briefing — Test prompt
*Generated 2026-05-26 · 2 sources · sentiment: positive*

## TL;DR
Something happened [^1].

## Key themes
- Theme one [^1]
- Theme two [^2]

## Sentiment
**Positive** — strong coverage.

## Sources
[^1]: [Title 1](https://example.com/1) — example.com, 2026-05-20
[^2]: [Title 2](https://example.com/2) — example.com, 2026-05-21
"""


def test_deterministic_check_passes_valid_briefing():
    result = checks.deterministic_check(VALID_BRIEFING)
    assert result["passed"] is True
    assert result["score"] == result["max_score"]
    assert result["sentiment_class"] == "positive"


def test_deterministic_check_fails_missing_sections():
    broken = "# Briefing — Test\n\nbody with no structure"
    result = checks.deterministic_check(broken)
    assert result["passed"] is False
    assert result["score"] < result["max_score"]
    assert result["checks"]["has_tldr_section"] is False
    assert result["checks"]["has_sources_section"] is False


def test_deterministic_check_detects_invalid_sentiment():
    no_sentiment = VALID_BRIEFING.replace("**Positive**", "**Mixed**").replace("positive", "mixed")
    result = checks.deterministic_check(no_sentiment)
    assert result["sentiment_class"] is None
    assert result["checks"]["sentiment_in_enum"] is False


def test_llm_judge_parses_valid_json(mock_llm_queue):
    mock_llm_queue.append(make_response([make_text_block(
        '{"coherence": 4, "citation_discipline": 5, "sentiment_fit": 4, "note": "themes could be tighter"}'
    )]))

    out = checks.llm_judge(VALID_BRIEFING, prompt="Test prompt")

    assert out["coherence"] == 4
    assert out["citation_discipline"] == 5
    assert out["sentiment_fit"] == 4
    assert "themes" in out["note"]


def test_llm_judge_handles_code_fence_wrapping(mock_llm_queue):
    mock_llm_queue.append(make_response([make_text_block(
        '```json\n{"coherence": 3, "citation_discipline": 3, "sentiment_fit": 3, "note": "okay"}\n```'
    )]))

    out = checks.llm_judge(VALID_BRIEFING, prompt="Test prompt")

    assert out["coherence"] == 3


def test_llm_judge_returns_error_on_non_json(mock_llm_queue):
    mock_llm_queue.append(make_response([make_text_block("This is not JSON at all.")]))

    out = checks.llm_judge(VALID_BRIEFING, prompt="Test prompt")

    assert "error" in out
    assert "raw" in out


def test_evaluate_briefing_file_loads_from_disk(tmp_path, mock_llm_queue):
    p = tmp_path / "briefing.md"
    p.write_text(VALID_BRIEFING)
    mock_llm_queue.append(make_response([make_text_block(
        '{"coherence": 5, "citation_discipline": 5, "sentiment_fit": 5, "note": "clean"}'
    )]))

    result = checks.evaluate_briefing_file(p)

    assert result["file"] == "briefing.md"
    assert result["deterministic"]["passed"] is True
    assert result["llm_judge"]["coherence"] == 5


def test_evaluate_briefing_file_skips_judge_when_disabled(tmp_path):
    p = tmp_path / "briefing.md"
    p.write_text(VALID_BRIEFING)

    result = checks.evaluate_briefing_file(p, use_llm_judge=False)

    assert "llm_judge" not in result
    assert result["deterministic"]["passed"] is True
