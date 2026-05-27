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


def test_loop_auto_attaches_documents_when_summarize_omits_them(
    mocker, mock_llm_queue, temp_data_dir
):
    """U0: model calls summarize(prompt=...) WITHOUT documents → loop injects fetched docs.

    This is the auto-attach path that eliminates the summarize_missing_documents
    failure pattern that fired in 6 of 7 historical runs.
    """
    from agent import storage

    # Seed cache with two successfully-fetched docs
    storage.save_fetched("https://a.example", {
        "url": "https://a.example", "text": "content a",
        "metadata": {"title": "A", "source_domain": "a.example"},
    })
    storage.save_fetched("https://b.example", {
        "url": "https://b.example", "text": "content b",
        "metadata": {"title": "B", "source_domain": "b.example"},
    })

    # Spy on summarize.run to inspect what documents arg arrives
    summarize_spy = mocker.patch(
        "agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING}
    )
    # fetch.run hits the cache (existing behavior)
    mocker.patch(
        "agent.tools.fetch.run",
        side_effect=lambda url: storage.load_fetched(url) or {"error": "miss"},
    )

    mock_llm_queue.extend([
        make_response([
            make_tool_use_block("t1", "fetch", {"url": "https://a.example"}),
            make_tool_use_block("t2", "fetch", {"url": "https://b.example"}),
        ]),
        # Model calls summarize WITHOUT documents — the failure mode U0 fixes
        make_response([make_tool_use_block("t3", "summarize", {"prompt": "test"})]),
        make_response([make_text_block("Done.")]),
    ])

    out = loop.run("Test prompt")

    assert out["status"] == "complete"
    # The summarize call should have received the auto-attached documents
    call_args = summarize_spy.call_args
    docs = call_args.kwargs.get("documents") or (call_args.args[1] if len(call_args.args) > 1 else None)
    assert docs is not None and len(docs) == 2
    urls = {d["url"] for d in docs}
    assert urls == {"https://a.example", "https://b.example"}


def test_loop_auto_attaches_when_model_passes_empty_documents(
    mocker, mock_llm_queue, temp_data_dir
):
    """U0 (C1 fix): model passing `documents=[]` is semantically equivalent
    to omitting the key — auto-attach must still fire.

    The original guard `"documents" not in tool_input` missed this case; the
    model would get an empty-docs stub briefing instead of the fetched docs.
    Truthiness check (`not tool_input.get("documents")`) closes the gap.
    """
    from agent import storage

    storage.save_fetched("https://x.example", {
        "url": "https://x.example", "text": "x content",
        "metadata": {"title": "X", "source_domain": "x.example"},
    })

    summarize_spy = mocker.patch(
        "agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING}
    )
    mocker.patch(
        "agent.tools.fetch.run",
        side_effect=lambda url: storage.load_fetched(url) or {"error": "miss"},
    )

    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "fetch", {"url": "https://x.example"})]),
        # Model passes EMPTY documents list — was a footgun in the original guard
        make_response([make_tool_use_block("t2", "summarize", {"prompt": "test", "documents": []})]),
        make_response([make_text_block("Done.")]),
    ])

    loop.run("Test prompt")

    call_args = summarize_spy.call_args
    docs = call_args.kwargs.get("documents") or (call_args.args[1] if len(call_args.args) > 1 else None)
    # Auto-attach should have fired and replaced the empty list
    assert docs is not None and len(docs) == 1
    assert docs[0]["url"] == "https://x.example"


def test_loop_auto_attaches_when_model_passes_none_documents(
    mocker, mock_llm_queue, temp_data_dir
):
    """U0 (C1 fix): model passing `documents=None` must also trigger auto-attach."""
    from agent import storage

    storage.save_fetched("https://x.example", {
        "url": "https://x.example", "text": "x",
        "metadata": {"title": "X", "source_domain": "x.example"},
    })

    summarize_spy = mocker.patch(
        "agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING}
    )
    mocker.patch(
        "agent.tools.fetch.run",
        side_effect=lambda url: storage.load_fetched(url) or {"error": "miss"},
    )

    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "fetch", {"url": "https://x.example"})]),
        make_response([make_tool_use_block("t2", "summarize", {"prompt": "test", "documents": None})]),
        make_response([make_text_block("Done.")]),
    ])

    loop.run("Test prompt")

    call_args = summarize_spy.call_args
    docs = call_args.kwargs.get("documents") or (call_args.args[1] if len(call_args.args) > 1 else None)
    assert docs is not None and len(docs) == 1


def test_loop_does_not_auto_attach_when_model_passes_documents(
    mocker, mock_llm_queue, temp_data_dir
):
    """U0: if the model passes documents explicitly, the loop does NOT override."""
    from agent import storage

    storage.save_fetched("https://cached.example", {
        "url": "https://cached.example", "text": "cached",
        "metadata": {"title": "C", "source_domain": "cached.example"},
    })

    summarize_spy = mocker.patch(
        "agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING}
    )
    mocker.patch(
        "agent.tools.fetch.run",
        side_effect=lambda url: storage.load_fetched(url) or {"error": "miss"},
    )

    explicit_docs = [{"url": "https://explicit.example", "text": "explicit"}]

    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "fetch", {"url": "https://cached.example"})]),
        make_response([make_tool_use_block(
            "t2", "summarize", {"prompt": "test", "documents": explicit_docs}
        )]),
        make_response([make_text_block("Done.")]),
    ])

    loop.run("Test prompt")

    call_args = summarize_spy.call_args
    docs = call_args.kwargs.get("documents") or (call_args.args[1] if len(call_args.args) > 1 else None)
    # The model's explicit docs should win — NOT the loop's auto-attach
    assert docs == explicit_docs


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
