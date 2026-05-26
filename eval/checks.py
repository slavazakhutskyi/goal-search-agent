"""Hybrid eval: deterministic schema checks + LLM-as-judge quality scoring.

This is a deliberately light harness — meant to demonstrate the pattern, not to be
a production eval framework. The grown-up version (labeled set + Cohen's kappa for
sentiment, regression suite, per-prompt rubrics, judge-quality calibration) is
named in README §"What I deliberately left out" and is the day-2 build.

Two layers:
- `deterministic_check(briefing)` — fast, free, runs on every briefing.
  Mirrors the schema asserts from tests/test_acceptance.py but at runtime so a
  freshly-generated briefing can be scored before commit / before delivery.
- `llm_judge(briefing, prompt)` — single Haiku call (~$0.0005) that scores the
  briefing on 3 axes (coherence, citation discipline, sentiment fit) on a 1-5
  scale. Haiku as judge is deliberately cheap; biases of using a smaller model
  to judge a larger model's output are real and worth flagging — see notes below.
"""

import json
import re
from pathlib import Path

from agent import llm

JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_PROMPT = """You are evaluating a competitive-intelligence briefing produced by an AI agent for an operator at a public safety AI company.

Original user prompt:
{prompt}

Briefing produced by the agent:
---
{briefing}
---

Score the briefing on three axes, 1-5 each:
- coherence: does the TL;DR + themes + mentions tell one consistent story?
- citation_discipline: does every factual claim have a [^N] footnote pointing to a source in the Sources section?
- sentiment_fit: does the sentiment verdict (positive/neutral/negative) actually match the tone of the coverage cited?

Return ONLY a single-line JSON object, no preamble, no commentary:
{{"coherence": <1-5>, "citation_discipline": <1-5>, "sentiment_fit": <1-5>, "note": "<one short sentence on the weakest axis>"}}
"""


def deterministic_check(briefing: str) -> dict:
    """Schema-shape validation. Returns a scorecard dict. No API calls."""
    has_header = briefing.startswith("# Briefing")
    has_tldr = "## TL;DR" in briefing
    has_themes = "## Key themes" in briefing
    has_sources = "## Sources" in briefing
    has_footnote = bool(re.search(r"\[\^\d+\]", briefing))
    sentiment_match = re.search(r"\b(positive|neutral|negative)\b", briefing, re.IGNORECASE)
    sentiment = sentiment_match.group(1).lower() if sentiment_match else None
    sentiment_valid = sentiment in {"positive", "neutral", "negative"}

    checks = {
        "has_briefing_header": has_header,
        "has_tldr_section": has_tldr,
        "has_themes_section": has_themes,
        "has_sources_section": has_sources,
        "has_footnote_citations": has_footnote,
        "sentiment_in_enum": sentiment_valid,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sentiment_class": sentiment,
        "score": sum(checks.values()),
        "max_score": len(checks),
    }


def llm_judge(briefing: str, prompt: str) -> dict:
    """LLM-as-judge using Haiku. Single API call. Returns scorecard + raw judge note.

    Caveat: Haiku judging a Sonnet briefing introduces a known bias (the judge
    may flatter or miss subtle errors the bigger model produced). Documented;
    not mitigated here. Production would use a held-out human-labeled set to
    calibrate the judge's scores before trusting them as a regression signal.
    """
    user_message = JUDGE_PROMPT.format(prompt=prompt, briefing=briefing[:20_000])  # cap context

    response = llm.call_with_retry({
        "model": JUDGE_MODEL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": user_message}],
    })

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    text = text.strip()

    # Be generous about parsing — judges sometimes wrap JSON in code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        scores = json.loads(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"error": f"judge returned non-JSON: {exc}", "raw": text[:200]}

    return scores


def evaluate_briefing_file(path: Path, prompt: str | None = None, use_llm_judge: bool = True) -> dict:
    """Run both layers on a briefing file. `prompt` defaults to the title line."""
    text = path.read_text()
    if prompt is None:
        first_line = text.splitlines()[0] if text else ""
        prompt = first_line.replace("# Briefing — ", "").strip() or "unknown"

    out = {
        "file": str(path.name),
        "deterministic": deterministic_check(text),
    }
    if use_llm_judge:
        out["llm_judge"] = llm_judge(text, prompt)
    return out
