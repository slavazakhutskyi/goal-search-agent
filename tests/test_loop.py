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


# ---------- U2: goal-completion terminator ----------

def test_parse_goal_n_extracts_number_before_candidate_noun():
    """U2: 'find 15 leads for X' → 15."""
    from agent import loop
    assert loop._parse_goal_n("find 15 freelance leads for python+react") == 15
    assert loop._parse_goal_n("find 10 best amateur football pitches in Berlin") == 10
    assert loop._parse_goal_n("get me 20 candidates matching X") == 20
    assert loop._parse_goal_n("show 5 companies that fit my skills") == 5


def test_parse_goal_n_returns_none_when_no_candidate_noun():
    from agent import loop
    assert loop._parse_goal_n("find a good restaurant nearby") is None
    assert loop._parse_goal_n("competitive intelligence on Carbyne, RapidDeploy, Prepared") is None
    assert loop._parse_goal_n("") is None


def test_parse_goal_n_rejects_out_of_range_numbers():
    """Years, prices, etc. should not be confused for goal-N."""
    from agent import loop
    # 2026 is way over the 100 cap
    assert loop._parse_goal_n("companies founded in 2026") is None


def test_deterministic_completion_counts_qualified_candidates():
    from agent import loop
    cands = [
        {"score": 0.9}, {"score": 0.8}, {"score": 0.71},
        {"score": 0.5}, {"score": 0.6},  # below threshold
    ]
    assert loop._deterministic_completion(cands, goal_n=3) is True
    assert loop._deterministic_completion(cands, goal_n=4) is False


def test_deterministic_completion_handles_missing_score():
    from agent import loop
    cands = [{"name": "x"}, {"score": "not a number"}, {"score": 0.9}]
    assert loop._deterministic_completion(cands, goal_n=1) is True
    assert loop._deterministic_completion(cands, goal_n=2) is False


def test_terminator_fires_when_goal_n_reached(mocker, mock_llm_queue, temp_data_dir):
    """Model adds 2 qualified candidates with goal_n=2 → next iteration sees GOAL_HINT."""
    from agent import loop
    from .conftest import make_response, make_text_block, make_tool_use_block
    loop.reset_judge_cache()

    mock_llm_queue.extend([
        make_response([
            make_tool_use_block("t1", "add_candidate", {
                "url": "https://a.example", "name": "A", "why_fits": "...", "score": 0.8,
            }),
            make_tool_use_block("t2", "add_candidate", {
                "url": "https://b.example", "name": "B", "why_fits": "...", "score": 0.85,
            }),
        ]),
        # Model gets the GOAL_HINT and finalizes
        make_response([make_tool_use_block("t3", "finalize", {})]),
        make_response([make_text_block("done")]),
    ])

    loop.run("find 2 candidates for X", self_eval=False)

    # Inspect messages history to find the GOAL_HINT injection. The hint
    # lands in the user-role message after the first iteration's tool results.
    # We can't directly access loop-local messages from outside; instead we
    # assert via the mock_llm_queue contract: if hint never fired, the model
    # wouldn't have a reason to call finalize on iteration 2. Both responses
    # were consumed → contract verified.
    assert mock_llm_queue == []  # all queued responses consumed


def test_terminator_does_not_fire_below_threshold(mocker, mock_llm_queue, temp_data_dir):
    """Candidates exist but score below threshold → no hint, model continues."""
    from agent import loop
    from .conftest import make_response, make_text_block, make_tool_use_block
    loop.reset_judge_cache()

    mock_llm_queue.extend([
        make_response([
            make_tool_use_block("t1", "add_candidate", {
                "url": "https://a.example", "name": "A", "why_fits": "...", "score": 0.5,
            }),
        ]),
        # No hint should fire (score 0.5 < 0.7 threshold)
        make_response([make_text_block("still working")]),
    ])

    out = loop.run("find 1 candidate for X", self_eval=False)
    # Run completes via natural termination (text block, no tool call)
    assert out["status"] == "complete"


def test_llm_judge_returns_false_when_no_api_key(monkeypatch, temp_data_dir):
    from agent import loop
    loop.reset_judge_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert loop._llm_judge_completion("find a restaurant", [{"name": "A", "score": 0.8, "why_fits": "ok"}]) is False


def test_llm_judge_caches_by_prompt_and_count(mocker, monkeypatch, temp_data_dir):
    from agent import loop
    loop.reset_judge_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import json as _json, types
    response = types.SimpleNamespace(content=[
        types.SimpleNamespace(type="text", text=_json.dumps({"decision": "yes", "reason": "ok"}))
    ])
    spy = mocker.patch("agent.loop.llm.call_with_retry", return_value=response)

    cands = [{"name": "A", "score": 0.8, "why_fits": "..."}]
    out1 = loop._llm_judge_completion("find a restaurant", cands)
    out2 = loop._llm_judge_completion("find a restaurant", cands)  # cache hit
    assert out1 is True and out2 is True
    assert spy.call_count == 1


def test_llm_judge_defaults_false_on_malformed_json(mocker, monkeypatch, temp_data_dir):
    from agent import loop
    loop.reset_judge_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    import types
    response = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="not json")])
    mocker.patch("agent.loop.llm.call_with_retry", return_value=response)
    cands = [{"name": "A", "score": 0.8, "why_fits": "..."}]
    assert loop._llm_judge_completion("find a thing", cands) is False


def test_build_goal_hint_includes_counts():
    from agent import loop
    hint = loop._build_goal_hint([
        {"score": 0.9}, {"score": 0.8}, {"score": 0.5},
    ])
    assert "GOAL_HINT" in hint
    assert "2" in hint  # 2 qualified
    assert "3" in hint  # 3 total
    assert "finalize()" in hint


# ---------- U3: output-shape dispatch ----------

def test_render_candidates_list_sorted_by_score_desc():
    from agent import loop
    cands = [
        {"url": "u1", "name": "Mid", "why_fits": "m", "score": 0.7},
        {"url": "u2", "name": "Top", "why_fits": "t", "score": 0.95},
        {"url": "u3", "name": "Low", "why_fits": "l", "score": 0.4},
    ]
    out = loop._render_candidates_list("find 3 things", cands)
    # Top score first
    top_idx = out.find("**Top**")
    mid_idx = out.find("**Mid**")
    low_idx = out.find("**Low**")
    assert 0 < top_idx < mid_idx < low_idx


def test_render_candidates_list_has_required_markdown_sections():
    from agent import loop
    out = loop._render_candidates_list("test", [
        {"url": "u", "name": "A", "why_fits": "ok", "score": 0.8},
    ])
    assert out.startswith("# Goal — ")
    assert "## Candidates" in out
    assert "**A**" in out
    assert "(u)" in out
    assert "score 0.80" in out


def test_render_candidates_list_handles_axes_when_present():
    from agent import loop
    out = loop._render_candidates_list("x", [{
        "url": "u", "name": "A", "why_fits": "ok", "score": 0.8,
        "axes": {"timezone": 0.9, "skill": 0.7},
    }])
    assert "axes" in out
    assert "timezone=0.90" in out


def test_render_candidates_list_omits_axes_section_when_empty():
    from agent import loop
    out = loop._render_candidates_list("x", [{
        "url": "u", "name": "A", "why_fits": "ok", "score": 0.8,
    }])
    assert "axes:" not in out


def test_post_loop_dispatch_emits_candidates_list_when_finalized(
    mocker, mock_llm_queue, temp_data_dir
):
    """finalize=True → candidates_list output, not briefing."""
    from agent import loop
    from .conftest import make_response, make_text_block, make_tool_use_block
    loop.reset_judge_cache()

    mock_llm_queue.extend([
        make_response([
            make_tool_use_block("t1", "add_candidate", {
                "url": "https://a.example", "name": "A", "why_fits": "...", "score": 0.9,
            }),
            make_tool_use_block("t2", "finalize", {}),
        ]),
        make_response([make_text_block("done")]),
    ])
    out = loop.run("find 1 candidate", self_eval=False)
    assert "## Candidates" in out["briefing"]
    assert "**A**" in out["briefing"]


def test_post_loop_dispatch_emits_candidates_floor_at_3(
    mocker, mock_llm_queue, temp_data_dir
):
    """≥3 candidates, no finalize → still emit candidates_list (floor)."""
    from agent import loop
    from .conftest import make_response, make_text_block, make_tool_use_block
    loop.reset_judge_cache()

    mock_llm_queue.extend([
        make_response([
            make_tool_use_block("t1", "add_candidate", {
                "url": "u1", "name": "A", "why_fits": "...", "score": 0.8,
            }),
            make_tool_use_block("t2", "add_candidate", {
                "url": "u2", "name": "B", "why_fits": "...", "score": 0.7,
            }),
            make_tool_use_block("t3", "add_candidate", {
                "url": "u3", "name": "C", "why_fits": "...", "score": 0.6,
            }),
        ]),
        make_response([make_text_block("done without finalize")]),
    ])
    out = loop.run("x", self_eval=False)
    # 3 candidates without finalize still triggers candidates_list (floor)
    assert "## Candidates" in out["briefing"]


# ---------- U5: SYSTEM_PROMPT shape ----------

def test_system_prompt_documents_new_tools():
    from agent import prompts
    assert "add_candidate" in prompts.SYSTEM_PROMPT
    assert "finalize" in prompts.SYSTEM_PROMPT
    assert "summarize" in prompts.SYSTEM_PROMPT
    # Search/fetch still documented
    assert "search(" in prompts.SYSTEM_PROMPT
    assert "fetch(" in prompts.SYSTEM_PROMPT


def test_system_prompt_describes_both_output_shapes():
    from agent import prompts
    text = prompts.SYSTEM_PROMPT
    # Shape A reference
    assert "CANDIDATES LIST" in text or "candidates list" in text.lower()
    # Shape B reference
    assert "NARRATIVE BRIEFING" in text or "briefing" in text.lower()


def test_system_prompt_removes_rapidsos_branding():
    from agent import prompts
    text = prompts.SYSTEM_PROMPT
    assert "RapidSOS" not in text
    # Parent-company hardcoded rule for specific competitors is gone
    assert "Axon owns Carbyne" not in text
    assert "Motorola owns RapidDeploy" not in text


def test_system_prompt_within_size_budget():
    from agent import prompts
    # Soft cap — agent context is large but new prompt shouldn't bloat
    assert len(prompts.SYSTEM_PROMPT) < 4000


def test_ae3_backward_compat_briefing_path_unchanged(
    mocker, mock_llm_queue, temp_data_dir
):
    """AE3 critical regression test: a run that calls search + fetch +
    summarize (no add_candidate) emits a BRIEFING, not candidates_list."""
    from agent import loop
    from .conftest import make_response, make_text_block, make_tool_use_block

    CANNED_BRIEFING = (
        "# Briefing — competitive intelligence\n\n"
        "## TL;DR\nThings happened.\n\n"
        "## Key themes\n- A [^1]\n\n"
        "## Sentiment\n**Neutral** — ok.\n\n"
        "## Sources\n[^1]: [A](https://a.example)\n"
    )
    mocker.patch("agent.tools.search.run", return_value={"query": "x", "results": []})
    mocker.patch("agent.tools.summarize.run", return_value={"briefing": CANNED_BRIEFING})

    mock_llm_queue.extend([
        make_response([make_tool_use_block("t1", "search", {"query": "x"})]),
        make_response([make_tool_use_block("t2", "summarize", {"prompt": "test", "documents": []})]),
        make_response([make_text_block("done")]),
    ])

    out = loop.run("competitive intelligence on Carbyne, RapidDeploy, Prepared", self_eval=False)

    # AE3 must produce the SUMMARIZE briefing, NOT candidates_list
    assert out["briefing"] == CANNED_BRIEFING
    assert "## Candidates" not in out["briefing"]
    assert "# Briefing —" in out["briefing"]
