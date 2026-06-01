"""Tests for the meta-eval module. Deterministic layers tested in isolation;
LLM-dependent layers tested via mocks."""

import json

import pytest

from agent import meta_eval, storage


# ---------- load_eval_corpus ----------

def test_load_eval_corpus_empty(temp_data_dir):
    out = meta_eval.load_eval_corpus()
    assert out == {"recent": [], "arc": {}, "total_runs": 0}


def test_load_eval_corpus_buckets_by_tag(temp_data_dir):
    records = [
        {"timestamp": "2026-05-26T10:00:00", "trace_issues": [{"tag": "X"}, {"tag": "Y"}]},
        {"timestamp": "2026-05-26T11:00:00", "trace_issues": [{"tag": "X"}]},
        {"timestamp": "2026-05-26T12:00:00", "trace_issues": [{"tag": "X"}, {"tag": "Z"}]},
    ]
    path = storage.LOGS / "evals.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))

    out = meta_eval.load_eval_corpus()
    assert out["total_runs"] == 3
    assert out["arc"]["X"]["count"] == 3
    assert out["arc"]["Y"]["count"] == 1
    assert out["arc"]["Z"]["count"] == 1
    assert out["arc"]["X"]["first_seen"] == "2026-05-26T10:00:00"
    assert out["arc"]["X"]["last_seen"] == "2026-05-26T12:00:00"


def test_load_eval_corpus_recent_caps_at_5(temp_data_dir):
    records = [{"timestamp": f"2026-05-26T{h:02d}:00:00", "trace_issues": []} for h in range(8)]
    path = storage.LOGS / "evals.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))

    out = meta_eval.load_eval_corpus()
    assert len(out["recent"]) == 5
    assert out["total_runs"] == 8


# ---------- load_edit_history ----------

def test_load_edit_history_empty(temp_data_dir):
    assert meta_eval.load_edit_history() == []


def test_load_edit_history_round_trip(temp_data_dir):
    record = {"edit_id": "E1", "anchor": "X", "verdict": "regressed", "kept": False}
    path = meta_eval._edit_history_path()
    path.write_text(json.dumps(record) + "\n")
    assert meta_eval.load_edit_history() == [record]


def test_partition_edit_history_maps_canary_verdicts(temp_data_dir):
    history = [
        {"anchor": "A", "verdict": "canary_promoted", "kept": True},
        {"anchor": "B", "verdict": "canary_discarded", "kept": False},
        {"anchor": "C", "verdict": "regressed", "kept": False},
    ]
    out = meta_eval._partition_edit_history(history)
    assert len(out["improved"]) == 1 and out["improved"][0]["anchor"] == "A"
    assert len(out["discarded"]) == 1 and out["discarded"][0]["anchor"] == "B"
    assert len(out["regressed"]) == 1 and out["regressed"][0]["anchor"] == "C"


# ---------- propose_edits ----------

def test_propose_short_circuits_on_empty_state(temp_data_dir, mocker):
    spy = mocker.spy(meta_eval.llm, "call_with_retry")
    result = meta_eval.propose_edits()
    assert result["proposed_edits"] == []
    assert "no eval records" in result["note"]
    assert spy.call_count == 0  # never called LLM


def test_propose_no_llm_mode(temp_data_dir, monkeypatch):
    # Seed evals so we don't short-circuit on empty
    path = storage.LOGS / "evals.jsonl"
    path.write_text(json.dumps({"timestamp": "2026-05-26T10:00:00", "trace_issues": [{"tag": "T"}]}) + "\n")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    result = meta_eval.propose_edits(use_llm=False)
    assert result["proposed_edits"] == []
    assert "no-llm" in result["note"]


def test_propose_parses_clean_json_and_validates_anchors(temp_data_dir, mocker, monkeypatch):
    # Seed evals
    path = storage.LOGS / "evals.jsonl"
    path.write_text(json.dumps({"timestamp": "2026-05-26T10:00:00", "trace_issues": [{"tag": "T"}]}) + "\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    payload = {
        "patterns_observed": [{"tag": "T", "count": 1, "trend": "new", "interpretation": "..."}],
        "prior_attempts_considered": [],
        "proposed_edits": [
            {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "...", "rationale": "...", "addresses_pattern_tag": "T"}
        ],
    }
    import types
    response = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=json.dumps(payload))])
    mocker.patch("agent.meta_eval.llm.call_with_retry", return_value=response)

    result = meta_eval.propose_edits(use_llm=True)
    assert len(result["proposed_edits"]) == 1
    edit = result["proposed_edits"][0]
    # The real SYSTEM_PROMPT contains "Strategy:" once → valid
    assert edit["anchor_count"] == 1
    assert edit["anchor_valid"] is True


def test_propose_flags_invalid_anchor(temp_data_dir, mocker, monkeypatch):
    path = storage.LOGS / "evals.jsonl"
    path.write_text(json.dumps({"timestamp": "2026-05-26T10:00:00", "trace_issues": [{"tag": "T"}]}) + "\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    payload = {
        "patterns_observed": [],
        "prior_attempts_considered": [],
        "proposed_edits": [
            {"id": "E1", "type": "add", "anchor": "NONEXISTENT-STRING-XYZ", "new_text": "...", "rationale": "..."}
        ],
    }
    import types
    response = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=json.dumps(payload))])
    mocker.patch("agent.meta_eval.llm.call_with_retry", return_value=response)

    result = meta_eval.propose_edits(use_llm=True)
    edit = result["proposed_edits"][0]
    assert edit["anchor_count"] == 0
    assert edit["anchor_valid"] is False


def test_propose_handles_malformed_llm_json(temp_data_dir, mocker, monkeypatch):
    path = storage.LOGS / "evals.jsonl"
    path.write_text(json.dumps({"timestamp": "2026-05-26T10:00:00", "trace_issues": [{"tag": "T"}]}) + "\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    import types
    response = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="not json at all")])
    mocker.patch("agent.meta_eval.llm.call_with_retry", return_value=response)

    result = meta_eval.propose_edits(use_llm=True)
    assert "error" in result
    assert "non-JSON" in result["error"]


def test_propose_persists_to_last_json(temp_data_dir, mocker, monkeypatch):
    path = storage.LOGS / "evals.jsonl"
    path.write_text(json.dumps({"timestamp": "2026-05-26T10:00:00", "trace_issues": [{"tag": "T"}]}) + "\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    payload = {"patterns_observed": [], "prior_attempts_considered": [], "proposed_edits": []}
    import types
    response = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=json.dumps(payload))])
    mocker.patch("agent.meta_eval.llm.call_with_retry", return_value=response)

    meta_eval.propose_edits(use_llm=True)
    assert (meta_eval._meta_evals_dir() / "last.json").exists()
    loaded = meta_eval.load_last_meta_eval()
    assert loaded["proposed_edits"] == []


# ---------- U2: apply_edit ----------

def _fake_prompts_py(tmp_path):
    """Build a minimal prompts.py at the meta_eval-expected path with a known SYSTEM_PROMPT."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    content = '''"""Stub prompts module for tests."""
from datetime import date

SYSTEM_PROMPT = f"""You are an agent.

Strategy:
1. Plan queries.
2. Fetch sources.
3. Summarize.

Constraints:
- Be concise.
"""

SUMMARIZE_PROMPT = """short."""
'''
    (agent_dir / "prompts.py").write_text(content)
    return agent_dir / "prompts.py"


def test_apply_edit_in_memory_replace():
    text = "Hello WORLD\nMore"
    edit = {"type": "replace", "anchor": "WORLD", "new_text": "EARTH"}
    assert meta_eval._apply_edit_in_memory(edit, text) == "Hello EARTH\nMore"


def test_apply_edit_in_memory_delete():
    text = "Hello WORLD\nMore"
    edit = {"type": "delete", "anchor": "WORLD"}
    assert meta_eval._apply_edit_in_memory(edit, text) == "Hello \nMore"


def test_apply_edit_in_memory_add_inserts_after_anchor():
    text = "Hello WORLD\nMore"
    edit = {"type": "add", "anchor": "WORLD", "new_text": "FRESH"}
    out = meta_eval._apply_edit_in_memory(edit, text)
    assert out == "Hello WORLD\nFRESH\nMore"


def test_apply_edit_in_memory_rejects_missing_anchor():
    with pytest.raises(ValueError, match="not found"):
        meta_eval._apply_edit_in_memory({"type": "add", "anchor": "ZZZ", "new_text": "x"}, "abc")


def test_apply_edit_in_memory_rejects_non_unique_anchor():
    with pytest.raises(ValueError, match="not unique"):
        meta_eval._apply_edit_in_memory({"type": "replace", "anchor": "X", "new_text": "Y"}, "X and X")


def test_apply_edit_dry_run_does_not_mutate(temp_data_dir, mocker, monkeypatch):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    # Seed a meta-eval with one valid edit
    meta = {
        "timestamp": "x",
        "proposed_edits": [
            {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- Cite every claim.",
             "rationale": "r", "anchor_valid": True, "anchor_count": 1},
        ],
    }
    meta_eval._save_meta_eval(meta)

    before = fake_path.read_text()
    result = meta_eval.apply_edit("E1", yes=False)
    assert result["dry_run"] is True
    assert "diff" in result and "+" in result["diff"]
    assert fake_path.read_text() == before  # unchanged
    # No edit_history record opened
    assert not meta_eval._edit_history_path().exists()


def test_apply_edit_with_yes_mutates_and_opens_record(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    meta = {
        "timestamp": "x",
        "proposed_edits": [
            {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- Cite every claim.",
             "rationale": "r", "anchor_valid": True, "anchor_count": 1},
        ],
    }
    meta_eval._save_meta_eval(meta)

    result = meta_eval.apply_edit("E1", yes=True)
    assert result["applied"] is True
    assert "Cite every claim." in fake_path.read_text()
    # Snapshot exists with original content
    snap_path = result["snapshot_path"]
    snap_text = open(snap_path).read()
    assert "Cite every claim." not in snap_text  # snapshot is pre-edit
    # edit_history record opened with verdict=pending
    history = meta_eval.load_edit_history()
    assert len(history) == 1
    assert history[0]["verdict"] == "pending"
    assert history[0]["kept"] is None


def test_apply_edit_refuses_when_anchor_invalid(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    meta = {
        "timestamp": "x",
        "proposed_edits": [
            {"id": "E1", "type": "add", "anchor": "DOESNOTEXIST", "new_text": "...",
             "anchor_valid": False, "anchor_count": 0},
        ],
    }
    meta_eval._save_meta_eval(meta)

    result = meta_eval.apply_edit("E1", yes=True)
    assert result["applied"] is False
    assert "not unique" in result["error"] or "anchor" in result["error"]


def test_apply_edit_rejects_syntax_breaking_edit(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    # Edit that injects an unterminated string into prompts.py
    meta = {
        "timestamp": "x",
        "proposed_edits": [
            {"id": "E1", "type": "replace", "anchor": '"""',
             "new_text": "unterminated", "anchor_valid": False, "anchor_count": 4},
        ],
    }
    meta_eval._save_meta_eval(meta)
    result = meta_eval.apply_edit("E1", yes=True)
    # Either anchor-validation OR syntax-check should reject
    assert result["applied"] is False


# ---------- U2: revert / keep ----------

def test_revert_last_edit_restores_file_and_flips_kept(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    meta = {"timestamp": "x", "proposed_edits": [
        {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- New rule.",
         "rationale": "r", "anchor_valid": True, "anchor_count": 1},
    ]}
    meta_eval._save_meta_eval(meta)
    meta_eval.apply_edit("E1", yes=True)
    assert "New rule." in fake_path.read_text()

    result = meta_eval.revert_last_edit(note="bad idea")
    assert result["reverted"] is True
    assert "New rule." not in fake_path.read_text()
    history = meta_eval.load_edit_history()
    assert history[-1]["kept"] is False
    assert history[-1]["note"] == "bad idea"


def test_revert_refuses_when_kept_already_set(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    meta = {"timestamp": "x", "proposed_edits": [
        {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- X.",
         "anchor_valid": True, "anchor_count": 1},
    ]}
    meta_eval._save_meta_eval(meta)
    meta_eval.apply_edit("E1", yes=True)
    meta_eval.keep_last_edit()  # closes the record
    result = meta_eval.revert_last_edit()
    assert result["reverted"] is False
    assert "already has kept" in result["error"]


def test_keep_last_edit_marks_kept_true(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    meta = {"timestamp": "x", "proposed_edits": [
        {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- X.",
         "anchor_valid": True, "anchor_count": 1},
    ]}
    meta_eval._save_meta_eval(meta)
    meta_eval.apply_edit("E1", yes=True)
    result = meta_eval.keep_last_edit(note="looks good")
    assert result["kept"] is True
    history = meta_eval.load_edit_history()
    assert history[-1]["kept"] is True
    assert history[-1]["note"] == "looks good"


# ---------- U7: classify_canary (composite verdict — U5 contract) ----------

def test_classify_canary_promote_on_avg_above_threshold():
    """U5: avg composite_delta > PROMOTE_DELTA (0.05) → promote."""
    decision, _ = meta_eval.classify_canary({"type": "add"}, [
        {"composite_delta": 0.08}, {"composite_delta": 0.10}
    ])
    assert decision == "promote"


def test_classify_canary_discard_on_severe_regression():
    """U5: any single composite_delta < DISCARD_DELTA (-0.10) → discard."""
    decision, _ = meta_eval.classify_canary({"type": "replace"}, [
        {"composite_delta": 0.05}, {"composite_delta": -0.20}
    ])
    assert decision == "discard"


def test_classify_canary_gate_on_noise_band():
    """U5: avg within (-0.10, +0.05] without severe regression → gate."""
    decision, _ = meta_eval.classify_canary({"type": "add"}, [
        {"composite_delta": 0.03}, {"composite_delta": -0.02}
    ])
    assert decision == "gate"


def test_classify_canary_gate_on_flat():
    decision, _ = meta_eval.classify_canary({"type": "add"}, [
        {"composite_delta": 0.0}, {"composite_delta": 0.0}
    ])
    assert decision == "gate"


def test_classify_canary_always_discards_delete():
    decision, reason = meta_eval.classify_canary({"type": "delete"}, [
        {"composite_delta": 0.20}
    ])
    assert decision == "discard"
    assert "blast radius" in reason


def test_classify_canary_gate_on_empty_k_results():
    decision, _ = meta_eval.classify_canary({"type": "add"}, [])
    assert decision == "gate"


# ---------- U7: _select_canary_prompt ----------

def test_select_canary_prompt_picks_complete_run_with_cached_urls(temp_data_dir):
    # Cache two docs
    storage.save_fetched("https://a.example", {"url": "https://a.example", "text": "a"})
    storage.save_fetched("https://b.example", {"url": "https://b.example", "text": "b"})
    # Write a fake briefing citing them
    briefing_dir = temp_data_dir / "briefings"
    briefing_dir.mkdir(exist_ok=True)
    briefing_path = briefing_dir / "20260526-100000-test.md"
    briefing_path.write_text(
        "# B\n[^1]: [A](https://a.example) — a, 2026-01-01\n"
        "[^2]: [B](https://b.example) — b, 2026-01-02\n"
    )
    # Write a runs.jsonl entry pointing at it
    runs_path = storage.LOGS / "runs.jsonl"
    runs_path.write_text(json.dumps({
        "timestamp": "2026-05-26T10:00:00",
        "prompt": "test prompt",
        "status": "complete",
        "briefing_path": str(briefing_path.relative_to(temp_data_dir)),
    }) + "\n")

    prompt, urls = meta_eval._select_canary_prompt()
    assert prompt == "test prompt"
    assert set(urls) == {"https://a.example", "https://b.example"}


def test_select_canary_prompt_returns_none_when_nothing_cached(temp_data_dir):
    runs_path = storage.LOGS / "runs.jsonl"
    runs_path.write_text(json.dumps({
        "timestamp": "x", "prompt": "p", "status": "complete",
        "briefing_path": "nonexistent.md",
    }) + "\n")
    prompt, urls = meta_eval._select_canary_prompt()
    assert prompt is None
    assert urls == []


# ---------- U7: run_canary (mocked replay) ----------

def test_run_canary_promote_path(temp_data_dir, mocker):
    """Mock replay to return briefings that score higher on the candidate side."""
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    # Also need agent.prompts.SYSTEM_PROMPT for the canary baseline — let it be the live module
    from agent import prompts as live_prompts
    # Edit must apply cleanly to live SYSTEM_PROMPT for the test
    edit = {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- Cite every claim.",
            "anchor_valid": True, "anchor_count": 1}

    # Mock replay.replay_run to return predictable briefings
    baseline_brief = "# B\n\n## TL;DR\nx\n\n## Key themes\n- a [^1]\n\n## Sources\n[^1]: [a](u)"
    candidate_brief = (
        "# B\n\n## TL;DR\nx\n\n## Key themes\n- a [^1]\n- b [^2]\n- c [^3]\n\n"
        "## Sentiment\n**Positive** — ok.\n\n## Sources\n[^1]: [a](u)\n"
    )
    call_count = {"n": 0}
    def fake_replay(prompt, urls, system_override):
        call_count["n"] += 1
        # Even calls are baseline, odd are candidate (per loop in run_canary)
        return {"briefing": baseline_brief if call_count["n"] % 2 == 1 else candidate_brief,
                "status": "complete", "tokens": {"input": 100, "output": 50}}
    mocker.patch("agent.replay.replay_run", side_effect=fake_replay)

    result = meta_eval.run_canary(edit, k=1, canary_prompt="p", canary_urls=["u1", "u2"])
    assert result["decision"] == "promote"
    assert result["delta"] > 0
    # Canary record saved
    canaries = list(meta_eval._canary_dir().glob("*.json"))
    assert len(canaries) == 1


def test_run_canary_discard_on_delete(temp_data_dir, mocker):
    """type=delete short-circuits to discard regardless of canary."""
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    edit = {"id": "E2", "type": "delete", "anchor": "Strategy:",
            "anchor_valid": True, "anchor_count": 1}
    # replay should NOT be called
    spy = mocker.patch("agent.replay.replay_run")
    result = meta_eval.run_canary(edit, k=1, canary_prompt="p", canary_urls=["u1", "u2"])
    assert result["decision"] == "discard"
    assert "blast radius" in result["reason"]
    assert spy.call_count == 0


def test_run_canary_gate_when_no_canary_prompt(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    edit = {"id": "E3", "type": "add", "anchor": "Strategy:", "new_text": "...",
            "anchor_valid": True, "anchor_count": 1}
    # No canary_prompt + no runs.jsonl → can't select → gate
    result = meta_eval.run_canary(edit, k=1)
    assert result["decision"] == "gate"
    assert "no historical prompt" in result["reason"]


# ---------- U8: autonomous orchestrator ----------

def _stub_proposal_with_one_edit():
    return {
        "timestamp": "x",
        "patterns_observed": [{"tag": "T", "count": 1, "trend": "new", "interpretation": "..."}],
        "prior_attempts_considered": [],
        "proposed_edits": [
            {"id": "E1", "type": "add", "anchor": "Strategy:", "new_text": "- New rule.",
             "rationale": "r", "addresses_pattern_tag": "T",
             "anchor_valid": True, "anchor_count": 1},
        ],
    }


def test_autonomous_promotes_when_canary_promotes(temp_data_dir, mocker):
    """Happy path: propose → canary promote → apply, with prompts.py mutated.

    Stub uses the U5 composite schema (baseline_composite/candidate_composite/
    composite_delta + baseline_breakdown/candidate_breakdown). Prior version
    used the pre-U5 int-delta shape, which made autonomous() silently write
    None into edit_history.composite_score (review T8/AC-1).
    """
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value=_stub_proposal_with_one_edit())

    def stub_canary(edit, **kw):
        return {
            "edit_id": edit["id"],
            "decision": "promote",
            "reason": "composite improved by +0.200",
            "delta": 0.20,
            "baseline_composite": 0.50,
            "candidate_composite": 0.70,
            "k": 1,
            "k_results": [{
                "iteration": 0,
                "canary_prompt": "p",
                "baseline_composite": 0.50,
                "candidate_composite": 0.70,
                "composite_delta": 0.20,
                "baseline_breakdown": {"operator": 0.4, "structure": 1.0, "efficiency": 0.5, "errors": 1.0},
                "candidate_breakdown": {"operator": 0.8, "structure": 1.0, "efficiency": 0.5, "errors": 1.0},
                "baseline_tokens": {"input": 1000, "output": 100},
                "candidate_tokens": {"input": 1000, "output": 100},
                "baseline_op_tokens": {"input": 200, "output": 100},
                "candidate_op_tokens": {"input": 200, "output": 100},
            }],
            "metric_version": "composite-v1",
        }
    mocker.patch("agent.meta_eval.run_canary", side_effect=stub_canary)
    meta_eval._save_meta_eval(_stub_proposal_with_one_edit())

    result = meta_eval.autonomous(max_cycles=1, max_cost=10.0)
    assert "New rule." in fake_path.read_text()
    assert result["cycles_run"] == 1
    cycle = result["cycles"][0]
    assert cycle["canary"]["decision"] == "promote"
    assert cycle["apply"]["applied"] is True
    # edit_history record updated to canary_promoted with new composite fields
    history = meta_eval.load_edit_history()
    assert history[-1]["verdict"] == "canary_promoted"
    assert history[-1]["kept"] is True
    # New fields per U5 should NOT be None
    assert history[-1]["composite_score"] == 0.70
    assert history[-1]["composite_breakdown"] == {
        "operator": 0.8, "structure": 1.0, "efficiency": 0.5, "errors": 1.0,
    }
    assert history[-1]["metric_version"] == "composite-v1"


def test_autonomous_halts_on_canary_gate(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value=_stub_proposal_with_one_edit())
    mocker.patch("agent.meta_eval.run_canary", return_value={
        "edit_id": "E1", "decision": "gate", "reason": "contradictory", "delta": 0.0, "k_results": [],
    })
    meta_eval._save_meta_eval(_stub_proposal_with_one_edit())

    before = fake_path.read_text()
    result = meta_eval.autonomous(max_cycles=3, max_cost=10.0)
    # prompts.py untouched on gate
    assert fake_path.read_text() == before
    assert "canary_gate" in result["halt_reason"]
    assert result["cycles_run"] == 1


def test_autonomous_records_discard_and_increments_regressions(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value=_stub_proposal_with_one_edit())
    mocker.patch("agent.meta_eval.run_canary", return_value={
        "edit_id": "E1", "decision": "discard", "reason": "regressed", "delta": -2.0,
        "k_results": [{"baseline_tokens": {"input": 100, "output": 20},
                       "candidate_tokens": {"input": 100, "output": 20}}],
    })
    meta_eval._save_meta_eval(_stub_proposal_with_one_edit())

    result = meta_eval.autonomous(max_cycles=3, max_cost=10.0)
    history = meta_eval.load_edit_history()
    # First cycle: canary discards → history record opened
    # Second cycle: anchor_reproposal guard catches the repeat BEFORE canary fires
    # (correct behavior — we halt at the cheapest signal, not always at consecutive_regressions)
    assert any(s in result["halt_reason"] for s in ("anchor_reproposal", "consecutive_regressions"))
    assert history[0]["verdict"] == "canary_discarded"
    assert history[0]["kept"] is False


def test_autonomous_anchor_reproposal_guard(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    # Pre-seed edit_history with a discarded anchor matching our proposed edit
    meta_eval.record_edit_attempt({
        "anchor": "Strategy:", "verdict": "canary_discarded", "kept": False,
    })
    mocker.patch("agent.meta_eval.propose_edits", return_value=_stub_proposal_with_one_edit())
    # run_canary should NOT be called — guard catches reproposal first
    spy = mocker.patch("agent.meta_eval.run_canary")
    meta_eval._save_meta_eval(_stub_proposal_with_one_edit())

    result = meta_eval.autonomous(max_cycles=3, max_cost=10.0)
    assert "anchor_reproposal" in result["halt_reason"]
    assert spy.call_count == 0


def test_autonomous_halts_on_cost_cap(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value=_stub_proposal_with_one_edit())
    # Massive tokens → blow through tiny budget on first cycle
    mocker.patch("agent.meta_eval.run_canary", return_value={
        "edit_id": "E1", "decision": "promote", "reason": "ok", "delta": 1.0,
        "k_results": [{"baseline_tokens": {"input": 1_000_000, "output": 1_000_000},
                       "candidate_tokens": {"input": 1_000_000, "output": 1_000_000}}],
    })
    meta_eval._save_meta_eval(_stub_proposal_with_one_edit())

    result = meta_eval.autonomous(max_cycles=5, max_cost=0.01)
    # First cycle consumes tokens past cap, second cycle pre-check halts
    assert "cost_cap" in result["halt_reason"]


def test_autonomous_cost_includes_operator_sim_tokens(temp_data_dir, mocker):
    """Regression test for PERF-001/REL-004: operator_sim Haiku tokens were
    invisible to the cost cap. Construct a scenario where the replay tokens
    alone don't breach max_cost but the op_tokens push it over."""
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value=_stub_proposal_with_one_edit())
    # Tiny replay tokens — would NOT trip cost-cap alone.
    # Large op_tokens — DO push the total past the cap.
    mocker.patch("agent.meta_eval.run_canary", return_value={
        "edit_id": "E1", "decision": "promote", "reason": "ok", "delta": 1.0,
        "k_results": [{
            "baseline_tokens": {"input": 100, "output": 10},
            "candidate_tokens": {"input": 100, "output": 10},
            # 2M total Haiku tokens — at $1/$5 per 1M, ~$8 cost. Way over $0.10.
            "baseline_op_tokens": {"input": 1_000_000, "output": 500_000},
            "candidate_op_tokens": {"input": 1_000_000, "output": 500_000},
        }],
    })
    meta_eval._save_meta_eval(_stub_proposal_with_one_edit())

    result = meta_eval.autonomous(max_cycles=3, max_cost=0.10)
    # Second cycle pre-check halts because op_tokens now count toward total
    assert "cost_cap" in result["halt_reason"]
    # Verify the operator-sim tokens were actually counted (cost > $0.10)
    assert result["total_cost_estimate"] > 0.10


def test_autonomous_convergence_when_no_valid_edits(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value={
        "patterns_observed": [], "prior_attempts_considered": [], "proposed_edits": [],
    })
    result = meta_eval.autonomous(max_cycles=3, max_cost=10.0)
    assert "convergence" in result["halt_reason"]
    assert result["cycles_run"] == 1


def test_autonomous_writes_audit_log(temp_data_dir, mocker):
    fake_path = _fake_prompts_py(temp_data_dir)
    mocker.patch("agent.meta_eval._prompts_py_path", return_value=fake_path)
    mocker.patch("agent.meta_eval.propose_edits", return_value={
        "patterns_observed": [], "prior_attempts_considered": [], "proposed_edits": [],
    })
    result = meta_eval.autonomous(max_cycles=1, max_cost=10.0)
    audit = Path(result["audit_path"])
    assert audit.exists()
    text = audit.read_text()
    assert "Autonomous meta-eval session" in text
    assert "Halt reason" in text


# Import Path for the test above
from pathlib import Path


# ---------- status() — agent-native CLI surface ----------

def test_status_exposes_demonstrations_readiness(temp_data_dir):
    """Agent-native gap fix (M-03): status() must expose whether the
    demonstrations layer is firing, not just legacy recent_lessons.

    Previously status only returned recent_lessons (lessons-block format),
    which loop.py no longer uses. The dashboard was showing stale data while
    the system actually uses demonstrations_block.
    """
    result = meta_eval.status()
    # New fields must be present (even when no composite records exist)
    assert "demonstrations_ready" in result
    assert "n_composite_records" in result
    assert "n_good_examples" in result
    assert "n_bad_examples" in result
    # With empty edit_history, demonstrations are NOT ready
    assert result["demonstrations_ready"] is False
    assert result["n_composite_records"] == 0
    # Legacy field still present for backward compat
    assert "recent_lessons" in result


def test_score_cli_verb(temp_data_dir, monkeypatch, capsys, tmp_path):
    """Agent-native gap fix: python -m agent.meta_eval score <briefing>
    must score a single briefing offline without firing canary infrastructure."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # force no-llm path; deterministic
    briefing_path = tmp_path / "test_briefing.md"
    briefing_path.write_text(
        "# Briefing — Test\n\n## TL;DR\nThings happened.\n\n"
        "## Key themes\n- A [^1]\n- B [^2]\n- C [^3]\n\n"
        "## Sentiment\n**Neutral** — ok.\n\n## Sources\n[^1]: [A](u)\n"
    )

    exit_code = meta_eval.main(["score", str(briefing_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Output should be JSON with structure + operator_sim + composite
    out = json.loads(captured.out)
    assert "structure" in out
    assert "operator_sim" in out
    assert "composite" in out
    assert "composite" in out["composite"]  # nested: composite_score returns {composite, breakdown, ...}
