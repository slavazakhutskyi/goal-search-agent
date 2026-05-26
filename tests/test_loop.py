"""Tests for agent.loop. Mocks agent.llm.call_with_retry via mock_llm_queue."""

import json

from agent import loop
from tests.conftest import make_response, make_text_block, make_tool_use_block


CANNED_BRIEFING = (
    "# Briefing — Test\n\n"
    "## TL;DR\nSomething happened.\n\n"
    "## Key themes\n- A theme [^1]\n\n"
    "## Sentiment\n**Neutral** — calm coverage.\n\n"
    "## Sources\n[^1]: [Title](https://example.com) — example.com, 2026-05-20\n"
)


def test_loop_natural_termination(mocker, mock_llm_queue, temp_data_dir):
    """search → summarize → text. Loop terminates at text, extracts briefing from summarize result."""
    # Mock tools so they don't actually run
    mocker.patch("agent.tools.search.run", return_value={"query": "rapidsos", "results": []})
    mocker.patch("agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING})

    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "search", {"query": "rapidsos"})]),
        make_response([make_tool_use_block("t2", "summarize", {"prompt": "test", "documents": []})]),
        make_response([make_text_block("Done. See briefing above.")]),
    ])

    out = loop.run("Test prompt")

    assert out["status"] == "complete"
    assert out["iterations"] == 3
    assert "TL;DR" in out["briefing"]
    assert out["briefing"] == CANNED_BRIEFING


def test_loop_max_iterations_partial(mocker, mock_llm_queue, temp_data_dir):
    """Model never terminates → status partial, fallback summarize runs."""
    mocker.patch("agent.tools.search.run", return_value={"query": "x", "results": []})
    fallback_summarize = mocker.patch(
        "agent.loop.summarize.run", return_value={"briefing": "# Briefing — fallback\n\n## TL;DR\npartial"}
    )

    # Queue MAX_ITERATIONS tool_use responses → model never terminates
    for i in range(loop.MAX_ITERATIONS):
        mock_llm_queue.append(
            make_response([make_tool_use_block(f"t{i}", "search", {"query": "x"})])
        )

    out = loop.run("Test prompt that runs forever")

    assert out["status"] == "partial_max_iterations"
    assert out["iterations"] == loop.MAX_ITERATIONS
    assert "fallback" in out["briefing"]
    fallback_summarize.assert_called_once()  # fallback path triggered


def test_loop_unknown_tool(mocker, mock_llm_queue, temp_data_dir):
    """Unknown tool name → error tool_result returned to model; loop continues."""
    mocker.patch("agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING})

    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "no_such_tool", {"foo": "bar"})]),
        make_response([make_tool_use_block("t2", "summarize", {"prompt": "x", "documents": []})]),
        make_response([make_text_block("done")]),
    ])

    out = loop.run("Test prompt")

    assert out["status"] == "complete"  # didn't crash on unknown tool
    assert out["iterations"] == 3


def test_loop_logs_run_to_jsonl(mocker, mock_llm_queue, temp_data_dir):
    """After loop.run, runs.jsonl has a new line with prompt + status + iterations."""
    mocker.patch("agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING})
    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "summarize", {"prompt": "x", "documents": []})]),
        make_response([make_text_block("done")]),
    ])

    loop.run("Logged prompt")

    runs_path = temp_data_dir / "logs" / "runs.jsonl"
    assert runs_path.exists()
    lines = runs_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["prompt"] == "Logged prompt"
    assert entry["status"] == "complete"
    assert entry["iterations"] == 2
    assert "elapsed_seconds" in entry
    assert "tool_calls" in entry


def test_loop_persists_briefing_to_data(mocker, mock_llm_queue, temp_data_dir):
    """A markdown briefing file appears in data/briefings/ after run()."""
    mocker.patch("agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING})
    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "summarize", {"prompt": "x", "documents": []})]),
        make_response([make_text_block("done")]),
    ])

    out = loop.run("Persisted prompt")

    briefings = list((temp_data_dir / "briefings").glob("*.md"))
    assert len(briefings) == 1
    assert briefings[0].read_text() == CANNED_BRIEFING
    assert out["briefing_path"] == briefings[0]


def test_loop_llm_exception_logged_as_partial(mocker, mock_llm_queue, temp_data_dir):
    """Non-retryable LLM error → status=partial_llm_error, runs.jsonl logged, briefing written."""
    import anthropic

    mocker.patch(
        "agent.llm.call_with_retry",
        side_effect=anthropic.AuthenticationError(
            message="bad key", response=mocker.MagicMock(), body=None,
        ),
    )

    out = loop.run("Test prompt with auth failure")

    assert out["status"] == "partial_llm_error"
    assert out["briefing"]  # non-empty stub briefing
    # runs.jsonl was written despite the failure
    runs_path = temp_data_dir / "logs" / "runs.jsonl"
    assert runs_path.exists()
    entry = json.loads(runs_path.read_text().strip().split("\n")[-1])
    assert entry["status"] == "partial_llm_error"
    assert "AuthenticationError" in entry.get("llm_error", "")


def test_loop_partial_reuses_successful_summarize(mocker, mock_llm_queue, temp_data_dir):
    """When summarize ran successfully but cap was hit, reuse that briefing (no extra call)."""
    successful_briefing = "# Briefing — From summarize\n\n## TL;DR\nReal output."
    summarize_spy = mocker.patch(
        "agent.tools.summarize.run", return_value={"briefing": successful_briefing}
    )
    mocker.patch("agent.tools.search.run", return_value={"query": "x", "results": []})

    # Iter 1: summarize (sets last_summarize_briefing).
    # Iters 2-8: search (model keeps looping; never terminates).
    mock_llm_queue.append(
        make_response([make_tool_use_block("t0", "summarize", {"prompt": "x", "documents": []})])
    )
    for i in range(loop.MAX_ITERATIONS - 1):
        mock_llm_queue.append(
            make_response([make_tool_use_block(f"t{i+1}", "search", {"query": "x"})])
        )

    out = loop.run("Cap hit after successful summarize")

    assert out["status"] == "partial_max_iterations"
    assert out["briefing"] == successful_briefing
    # summarize.run was called exactly once (the original call, not a fallback)
    assert summarize_spy.call_count == 1


def test_loop_empty_response_marks_partial(mocker, mock_llm_queue, temp_data_dir):
    """Model returns response with no tool_use and no text → status=partial_empty_response."""
    mocker.patch("agent.tools.summarize.run", return_value={"briefing": "# Briefing — fallback\n\n## TL;DR\nok"})
    mock_llm_queue.append(make_response([]))  # empty content blocks

    out = loop.run("Test empty response")

    assert out["status"] == "partial_empty_response"
    # Fallback briefing exists (either from a successful summarize earlier or from summarize.run fallback)
    assert out["briefing"]
