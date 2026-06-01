"""Tests for the replay layer. Mocks the LLM; verifies search/fetch are
properly swapped + restored, and that the system prompt override threads
through to the loop."""

import json

import pytest

from agent import loop, prompts, replay, storage
from agent.tools import fetch, search

from .conftest import make_response, make_text_block, make_tool_use_block


# ---------- cached_mode context manager ----------

def test_cached_mode_swaps_and_restores_search_and_fetch(temp_data_dir):
    """search.run + fetch.run + prompts.SYSTEM_PROMPT all restore on exit."""
    orig_search = search.run
    orig_fetch = fetch.run
    orig_prompt = prompts.SYSTEM_PROMPT

    with replay.cached_mode([], system_override="ALT PROMPT"):
        assert search.run is not orig_search
        assert fetch.run is not orig_fetch
        assert prompts.SYSTEM_PROMPT == "ALT PROMPT"

    assert search.run is orig_search
    assert fetch.run is orig_fetch
    assert prompts.SYSTEM_PROMPT is orig_prompt


def test_cached_mode_restores_on_exception(temp_data_dir):
    orig_search = search.run
    orig_fetch = fetch.run

    with pytest.raises(RuntimeError, match="boom"):
        with replay.cached_mode([], system_override=None):
            raise RuntimeError("boom")

    assert search.run is orig_search
    assert fetch.run is orig_fetch


def test_mock_search_returns_only_allowed_urls(temp_data_dir):
    """search.run inside cached_mode yields exactly the cached docs we allowed."""
    storage.save_fetched("https://a.example", {
        "url": "https://a.example", "text": "doc a text",
        "metadata": {"title": "A title", "source_domain": "a.example"},
    })
    storage.save_fetched("https://b.example", {
        "url": "https://b.example", "text": "doc b text",
        "metadata": {"title": "B title", "source_domain": "b.example"},
    })
    # Allow only A
    with replay.cached_mode(["https://a.example"], system_override=None):
        result = search.run("any query")
        urls = [r["url"] for r in result["results"]]
        assert urls == ["https://a.example"]
        assert result["results"][0]["title"] == "A title"


def test_mock_fetch_refuses_disallowed_url(temp_data_dir):
    """fetch.run inside cached_mode errors on URLs outside the allowed set."""
    storage.save_fetched("https://a.example", {
        "url": "https://a.example", "text": "doc a",
        "metadata": {"title": "A", "source_domain": "a.example"},
    })
    with replay.cached_mode(["https://a.example"], system_override=None):
        ok = fetch.run("https://a.example")
        bad = fetch.run("https://outside.example")
        assert ok.get("text") == "doc a"
        assert "not in canary cache" in (bad.get("error") or "")


def test_mock_search_skips_cached_docs_with_errors(temp_data_dir):
    """Docs cached with `error` flag (e.g., 403 attempt) are excluded."""
    storage.save_fetched("https://err.example", {
        "url": "https://err.example", "error": "403", "text": "",
        "metadata": {"source_domain": "err.example"},
    })
    storage.save_fetched("https://ok.example", {
        "url": "https://ok.example", "text": "good text",
        "metadata": {"title": "OK", "source_domain": "ok.example"},
    })
    with replay.cached_mode(["https://err.example", "https://ok.example"], system_override=None):
        result = search.run("q")
        urls = [r["url"] for r in result["results"]]
        assert urls == ["https://ok.example"]


# ---------- replay_run end-to-end ----------

def test_replay_run_uses_overridden_system_prompt(temp_data_dir, mock_llm_queue, mocker):
    """The system_override actually reaches the Anthropic call inside the loop."""
    # Cache one doc so the model has something to "search" for
    storage.save_fetched("https://x.example", {
        "url": "https://x.example", "text": "x content",
        "metadata": {"title": "X", "source_domain": "x.example"},
    })
    # Model returns a single text response, no tool calls → loop exits as complete
    mock_llm_queue.append(make_response([make_text_block("done briefing")]))

    # Spy on llm.call_with_retry to inspect the system kwarg
    spy = mocker.spy(loop.llm, "call_with_retry")

    result = replay.replay_run(
        "test prompt",
        allowed_urls=["https://x.example"],
        system_override="OVERRIDDEN SYSTEM",
    )

    assert result["status"] == "complete"
    # First call's kwargs carry the overridden system prompt
    first_call_kwargs = spy.call_args_list[0][0][0]
    assert first_call_kwargs["system"].startswith("OVERRIDDEN SYSTEM")


def test_replay_run_skips_self_eval(temp_data_dir, mock_llm_queue):
    """Replays must NOT write to evals.jsonl — they aren't canonical runs."""
    storage.save_fetched("https://x.example", {
        "url": "https://x.example", "text": "x",
        "metadata": {"title": "X", "source_domain": "x.example"},
    })
    mock_llm_queue.append(make_response([make_text_block("done")]))

    replay.replay_run("p", ["https://x.example"], system_override="S")

    evals_path = storage.LOGS / "evals.jsonl"
    # File should not exist OR should be empty
    assert not evals_path.exists() or evals_path.read_text().strip() == ""
