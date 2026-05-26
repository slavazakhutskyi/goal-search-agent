"""Per-prompt schema-shape behavior tests (formerly U5, folded into U4).

Validates that the agent end-to-end produces a structurally-valid briefing for
each of the 4 spec prompts. Mocks the Anthropic client only — search and fetch
are mocked via httpx-level patches in the loop tests, but here we shortcut:
the Anthropic client returns text-only (no tool_use), so the loop terminates
on iteration 1 using the canned briefing as the final answer.

This is schema validation only. Content correctness is U6's manual concern
(real briefings generated against real Anthropic + real SearXNG).
"""

import re

from agent import loop
from tests.conftest import make_response, make_text_block


def _canned_briefing(topic: str, sentiment: str, surface: str = "") -> str:
    """Build a schema-valid briefing string for tests. surface = extra body text."""
    return (
        f"# Briefing — {topic}\n"
        f"*Generated 2026-05-26T12:00:00+00:00 · 3 sources · sentiment: {sentiment.lower()}*\n\n"
        f"## TL;DR\nA short summary of {topic}. {surface}\n\n"
        f"## Key themes\n"
        f"- Theme one [^1]\n"
        f"- Theme two [^2]\n\n"
        f"## Notable mentions\n"
        f"> \"A quote\" — Source [^1]\n\n"
        f"## Sentiment\n**{sentiment.capitalize()}** — justification.\n\n"
        f"## Sources\n"
        f"[^1]: [Title 1](https://example.com/1) — example.com, 2026-05-20\n"
        f"[^2]: [Title 2](https://example.com/2) — example.com, 2026-05-21\n"
    )


def assert_valid_briefing(markdown: str, expected_sentiment_in: set[str] | None = None):
    """Schema-shape assertions only. Never asserts specific content."""
    assert markdown.startswith("# Briefing"), "missing '# Briefing' header"
    assert "## TL;DR" in markdown, "missing TL;DR section"
    assert "## Key themes" in markdown, "missing Key themes section"
    assert "## Sources" in markdown, "missing Sources section"
    # at least one footnote reference
    assert re.search(r"\[\^\d+\]", markdown), "no footnote citations found"
    # sentiment is one of the three valid values
    sentiments = {"positive", "neutral", "negative"}
    found = {s for s in sentiments if re.search(rf"\b{s}\b", markdown, re.IGNORECASE)}
    assert found, "no sentiment class found"
    if expected_sentiment_in is not None:
        assert found & expected_sentiment_in, f"sentiment {found} not in expected {expected_sentiment_in}"


def test_prompt1_rapidsos_7day(mock_llm_queue, temp_data_dir):
    """Covers AE1. Spec prompt #1: RapidSOS coverage past 7 days."""
    briefing = _canned_briefing("RapidSOS in the last 7 days", "positive")
    mock_llm_queue.append(make_response([make_text_block(briefing)]))

    out = loop.run("Give me a briefing on everything published about RapidSOS in the last 7 days.")

    assert_valid_briefing(out["briefing"])
    assert out["status"] == "complete"


def test_prompt2_top_3_stories(mock_llm_queue, temp_data_dir):
    """Spec prompt #2: top 3 public safety AI stories. (No dedicated origin AE — covers R1.)"""
    briefing = _canned_briefing("top 3 public safety AI stories this week", "neutral")
    mock_llm_queue.append(make_response([make_text_block(briefing)]))

    out = loop.run("What are the top 3 public safety AI stories from this week?")

    assert_valid_briefing(out["briefing"])


def test_prompt3_competitors_with_parents(mock_llm_queue, temp_data_dir):
    """Covers AE2. Spec prompt #3: competitors. Briefing MUST surface parent companies."""
    # Realistic canned content surfacing parent companies per origin §AE2
    briefing = _canned_briefing(
        topic="competitor coverage",
        sentiment="neutral",
        surface="Carbyne (Axon) and Prepared (Axon) saw integration progress; RapidDeploy (Motorola) shipped a feature.",
    )
    mock_llm_queue.append(make_response([make_text_block(briefing)]))

    out = loop.run("Find any press releases or news mentions of our competitors: Carbyne, RapidDeploy, Prepared.")

    assert_valid_briefing(out["briefing"])
    body = out["briefing"]
    # Parent-company surfacing requirement
    assert any(name in body for name in ("Carbyne", "Prepared", "RapidDeploy")), \
        "no competitor name in briefing"
    assert any(parent in body for parent in ("Axon", "Motorola")), \
        "no parent-company surfacing (Axon/Motorola)"


def test_prompt4_sentiment_3class(mock_llm_queue, temp_data_dir):
    """Covers AE3. Spec prompt #4: sentiment, must be one of 3 classes."""
    briefing = _canned_briefing("sentiment on AI in emergency dispatch", "neutral")
    mock_llm_queue.append(make_response([make_text_block(briefing)]))

    out = loop.run("Summarize the sentiment of recent coverage of AI in emergency dispatch.")

    assert_valid_briefing(out["briefing"], expected_sentiment_in={"positive", "neutral", "negative"})
