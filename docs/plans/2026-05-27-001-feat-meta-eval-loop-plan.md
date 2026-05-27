# Plan — Meta-eval loop with hybrid canary-replay autonomy

> Created: 2026-05-27 · Updated: 2026-05-27 (hybrid canary extension) · Depth: Deep
> Source: live-interview prep session — brainstorm + timeout + ce-plan, in-thread
> Parent feature: `agent/evaluate.py` (per-run critique + lessons feedback shipped earlier this session)
> Mode: **hybrid**. Manual verbs available; primary path is autonomous with canary-replay safety nets and outcome-gated human review (fires only on contradictory canary results).

## Goal

Close the loop between per-run lessons and the static `SYSTEM_PROMPT`, with a **compounding-knowledge layer** so the system learns from its own past learning attempts. After each batch of runs, an LLM-driven meta-eval reads THREE inputs — recent lessons, the full lesson arc, AND the edit history — and proposes concrete edits to `agent/prompts.py:SYSTEM_PROMPT`. A human approves; the edit is applied with snapshot; the same historical prompt is re-run; a deterministic + side-by-side comparison report is emitted; the verdict is recorded back into the edit history. Cycle is repeatable AND each cycle has access to every prior cycle's outcome.

**Compounding mechanism (the key insight):**
- Recent lessons (last 5) → short-term signal: what's failing NOW
- Full lesson arc → chronic vs fresh distinction: which failures persist across runs
- Edit history → meta-knowledge: what we've TRIED to fix, what worked, what regressed
- Meta-eval prompt explicitly: *"Do NOT re-propose edits that regressed previously. DO propose variations on improvements. Patterns chronic for N runs with no successful edit need a different angle."*

## Hybrid autonomy model (the key architectural addition)

Two execution paths share the same primitives (`propose`, `apply`, `compare`, `keep`, `revert`):

**Manual path** (U1-U5): human runs each verb individually. Useful for debugging, fine inspection, and as the fallback when autonomous halts.

**Autonomous path** (U6-U9): one command — `python -m agent.meta_eval autonomous --prompt "<historical prompt substring>"` — runs the full cycle without per-step approval. Safety comes from:

1. **Canary replay before any `prompts.py` mutation.** Each proposed edit applies to a TEMP system-prompt overlay, runs K=3 mini-replays (only the `summarize` step, using cached fetched docs — no new search/fetch cost), computes structure scores.
2. **Deterministic canary verdict.** All 3 mini-runs ≥ baseline AND no run regressed by >1 point → `promote` (apply to real `prompts.py`). Any drop > 1 → `discard` (edit never touches the file). Mixed/contradictory results → `gate` (halt loop, surface for human review).
3. **Session-level safety nets.** Cost cap $0.50; cycle cap 5; 2 consecutive in-session regressions → halt; anchor reproposed from `edit_history.regressed[*]` → halt + surface (meta-eval prompt is failing its own discipline rule).
4. **Per-edit reversibility.** Every promoted edit snapshots `prompts.py` first. Worst case = one bad canary-passing edit, instantly revertable via `revert_last_edit()`.
5. **Audit trail.** `data/logs/autonomous_runs/{ts}.md` captures every cycle: input lessons, proposed edits, canary scores, decisions, cumulative cost. Human reviews AFTER, not during.

**Defense story** (the answer to "what stops this from drifting"):
> "Edits never touch the live file until they survive 3 canary replays against historical prompts using cached web data. Cost is bounded. Convergence and consecutive-regression detectors halt the loop. The human gate fires only when canary results are contradictory — the human's attention goes where judgment is actually needed."

## Non-goals

- Editing `SUMMARIZE_PROMPT` (separate surface; this cycle only touches `SYSTEM_PROMPT`)
- Auto-promoting based on canary alone for high-blast-radius edits (`type=delete` always gates regardless of canary outcome — explicit override in U7)
- Statistical significance from K=3 (this is heuristic safety, not proof; production version would use K=10+ with hypothesis testing)
- Cross-session learning of which CANARY decisions were correct (deferred — would require post-hoc human labels)

## Architecture choice

Two new modules:
- **`agent/meta_eval.py`** — orchestrator with verbs `propose`, `apply`, `compare`, `keep`, `revert`, `status`, `autonomous`. Reuses `agent.evaluate` for structure scoring and `agent.loop.run()` for full re-execution.
- **`agent/replay.py`** — cached-doc replay layer. Re-runs ONLY the `summarize` step using documents already in `data/fetched/`, against a swap-in alternate `SYSTEM_PROMPT`. ~50 lines. The canary mechanism's foundation; also unlocks fast offline prompt iteration.

Persistence:
- `data/logs/meta_evals/{ts}.json` — Sonnet meta-eval outputs (proposed edits + patterns observed + prior-attempts considered)
- `data/logs/prompts_history/{ts}-pre-edit.py` — `prompts.py` snapshots before each apply
- `data/logs/comparisons/{ts}-{slug}.md` — before/after comparison reports
- `data/logs/canary/{ts}-E{id}.json` — per-canary mini-run scores + decision (promote/discard/gate)
- `data/logs/edit_history.jsonl` — every applied edit + lifecycle (verdict, kept/reverted)
- `data/logs/autonomous_runs/{ts}.md` — per-session audit trail

Why two modules: `replay.py` is small but distinct in purpose (pure replay logic, no LLM judgment) and is reused outside autonomous mode (manual offline prompt iteration). Mixing into `meta_eval.py` would couple unrelated concerns.

Why hybrid not pure-manual: the brainstorm rejected pure-autonomous (rule drift, reward hacking, cost spiral) AND pure-manual (too slow to demonstrate compounding in interview Part 2). Hybrid puts the human gate at the moment judgment is actually required — contradictory canary signals — and auto-handles the unambiguous cases. See "Hybrid autonomy model" section above for the full rationale.

## Implementation units (9, dependency-ordered)

**U1-U5**: manual path (foundation + closing-the-loop verbs + manual demo).
**U6-U9**: hybrid autonomy path (replay + canary + autonomous orchestrator + autonomous demo).

The manual path is built first because the autonomous path REUSES its primitives (`apply`, `edit_history`). U6 (replay) can be built in parallel with U2-U4 since it has no dependency on them.

### U1 — `propose`: LLM-driven SYSTEM_PROMPT improvement suggestions (THREE-input compounding)

**Goal.** Read all THREE knowledge inputs — recent lessons, full lesson arc, edit history — send to Sonnet, receive structured proposed edits as JSON. The meta-eval is explicitly told what's been tried before so it doesn't repeat failures.

**Files.**
- `agent/meta_eval.py` (new): `load_eval_corpus()`, `load_edit_history()`, `META_EVAL_PROMPT`, `propose_edits()`
- `data/logs/meta_evals/` (new dir, gitignored beyond `.gitkeep`)

**Approach.** `load_eval_corpus()` reads `evals.jsonl` and partitions it:
- **`recent`**: last 5 records (verbatim) — high signal, what's failing NOW
- **`arc`**: ALL records bucketed by `trace_issues[*].tag` with counts and first/last-seen timestamps — chronic vs fresh

`load_edit_history()` reads `data/logs/edit_history.jsonl` (created by U2) — every prior applied edit with: anchor, new_text, rationale, before/after structure scores, verdict ∈ {improved, regressed, inconclusive, pending}, kept ∈ {true, false, null}.

`propose_edits()` builds a structured payload combining all three, samples ≤3 briefing excerpts per distinct prompt type for grounding, and calls Sonnet (NOT Haiku — judgment quality matters) with `META_EVAL_PROMPT` that includes:
- The three knowledge buckets explicitly
- Hard rules: *"Do NOT propose any edit with an anchor that appears in `edit_history.regressed[*]`. DO consider variations on `edit_history.improved[*].anchor` for adjacent issues. For patterns with `count ≥ 3` and zero successful edits in history, propose a STRUCTURALLY different approach (e.g., new tool guidance vs. tightened existing rule)."*
- STRICT JSON output:

```json
{
  "patterns_observed": [{"tag": "...", "count": N, "trend": "chronic|new|resolved", "interpretation": "..."}],
  "prior_attempts_considered": [{"edit_anchor": "...", "verdict": "...", "decision": "skip|vary|build_on"}],
  "proposed_edits": [
    {"id": "E1", "type": "add|replace|delete", "anchor": "<exact substring in SYSTEM_PROMPT>",
     "new_text": "...", "rationale": "...", "addresses_pattern_tag": "..."}
  ]
}
```

Saves output to `data/logs/meta_evals/{ts}.json`. Prints a human-readable summary to stderr including which prior attempts informed which decisions.

**Test scenarios.**
1. Empty `evals.jsonl` AND empty `edit_history.jsonl` → returns empty result without calling LLM
2. Mocked LLM returns malformed JSON → function returns `{error: ...}` without crashing
3. With ≥1 eval record + ≥1 regressed edit history record, the input payload to the LLM contains the regressed edit in `edit_history.regressed[*]` (verified via spy)
4. Bucketing test: 3 eval records with same `trace_issues[*].tag` → `arc[tag].count == 3` and `recent` contains all 3

**Why riskiest.** Load-bearing for entire feature. If Sonnet produces vague / unactionable / non-anchored edits, or ignores the prior-attempts data, the compounding loop is broken. Empirical test in U6: after one cycle with a regressed edit, re-run `propose` and verify the LLM does NOT re-propose the same anchor.

---

### U2 — `apply`: edit `prompts.py` with snapshot + diff + edit_history record

**Goal.** Take one proposed edit, apply it to `agent/prompts.py`, snapshot pre-edit state, AND open a record in `edit_history.jsonl` with verdict=`pending` (to be filled by U3 and U4).

**Files.**
- `agent/meta_eval.py` (extend): `apply_edit()`, `snapshot_prompts()`, `revert_last_edit()`, `record_edit_attempt()`
- `data/logs/prompts_history/{ts}-pre-edit.py` (new, snapshot output)
- `data/logs/edit_history.jsonl` (new, append-only log of every applied edit + its lifecycle)
- `agent/prompts.py` (mutated when `--yes` passed)

**Approach.** `snapshot_prompts()` copies current `prompts.py` to `prompts_history/{ts}-pre-edit.py`. `apply_edit()` loads the proposed edits file, finds the edit by `id`, locates `anchor` in current `prompts.py`. Fails loudly if not unique. For `type=add`: insert after anchor. For `type=replace`: replace anchor. For `type=delete`: drop anchor. Print unified diff. Without `--yes`: dry-run. With `--yes`: write file AND call `record_edit_attempt()` which appends to `edit_history.jsonl`:

```json
{"timestamp": "...", "edit_id": "E1", "type": "add", "anchor": "...", "new_text": "...",
 "rationale": "...", "snapshot_path": "data/logs/prompts_history/...",
 "verdict": "pending", "before_score": null, "after_score": null, "kept": null}
```

`revert_last_edit()` restores from most recent snapshot and updates the trailing edit_history record with `kept: false`.

**Test scenarios.**
1. Dry-run prints diff, file byte-identical after call, NO edit_history record written
2. `--yes` writes file + snapshot file + edit_history record with `verdict=pending`
3. Anchor not found → clear error, no snapshot, no edit_history record (no orphans)
4. `revert_last_edit()` restores file content AND flips `kept: false` in the trailing edit_history record

---

### U3 — `compare`: re-run a historical prompt, score the delta, update edit_history verdict

**Goal.** Re-run a historical prompt against current (post-U2) `SYSTEM_PROMPT`. Produce a comparison report. Update the trailing `edit_history.jsonl` record with computed `before_score`, `after_score`, and `verdict`.

**Files.**
- `agent/meta_eval.py` (extend): `compare()`, `_render_comparison_md()`, `update_edit_verdict()`
- `data/logs/comparisons/{ts}-{slug}.md` (new, comparison report)
- `data/logs/edit_history.jsonl` (updated — trailing record's verdict + scores filled in)

**Approach.** `compare(prompt_substring)` finds the most recent matching run in `runs.jsonl`, loads its briefing as the BEFORE. Calls `loop.run(prompt, self_eval=False)` to produce the AFTER briefing (cost: ~$0.05-0.10 per cycle). Runs `evaluate.check_structure()` on both. Renders markdown report with: before/after table (structure score, themes, citations, fail list, tool calls, iterations, tokens, cost), then both briefings inlined for human side-by-side reading.

Verdict heuristic (deterministic, no LLM):
- `improved`: structure score up AND tool-call count not up by >50%
- `regressed`: structure score down OR tool-call count up by >100%
- `inconclusive`: structure score unchanged

`update_edit_verdict()` opens `edit_history.jsonl`, finds the trailing `pending` record, rewrites it in-place with: `before_score`, `after_score`, `verdict`. `kept` stays null until U4.

**Test scenarios.**
1. Mocked `loop.run` returns a briefing → comparison report written + trailing edit_history record has verdict filled
2. Verdict heuristic boundary cases: +1 score same tools → `improved`; -1 score → `regressed`; +0 → `inconclusive`
3. No pending edit_history record → compare still runs (for one-off re-runs without a preceding apply) but skips the update
4. Substring matches multiple runs → uses most recent (verified via timestamp ordering)

---

### U4 — CLI + closing-the-loop verbs (`keep`, `revert`, `status`)

**Goal.** Wire all verbs into `python -m agent.meta_eval` and add the closing-the-loop pair: `keep` (commits the edit) and `revert` (restores from snapshot). Both write the final `kept` field on the trailing `edit_history.jsonl` record so the NEXT meta-eval cycle can learn from this attempt.

**Files.**
- `agent/meta_eval.py` (extend): `keep_last_edit()`, `revert_last_edit()`, `status()`, `main()`, `if __name__ == "__main__"`
- `data/logs/edit_history.jsonl` (updated by `keep`/`revert`)
- `agent/prompts.py` (restored on `revert`)

**Approach.** `keep_last_edit()` operates on the trailing record in `edit_history.jsonl`: sets `kept=true`, appends optional `--note`. `revert_last_edit()` reads trailing record's `snapshot_path`, restores `agent/prompts.py` byte-for-byte from snapshot, sets `kept=false` + note. Both refuse to run if `kept` is already non-null (corruption guard). `status()` prints a compact dashboard: trailing edit_history record + recent lessons summary + which step in the cycle we're at.

`main()` dispatches on `argv[1]` ∈ {`propose`, `apply`, `compare`, `keep`, `revert`, `status`}. Per-verb argparse: `propose` takes `--no-llm`; `apply` takes `<edit_id>` + `--yes`; `compare` takes `<prompt_substring>`; `keep`/`revert` take optional `--note`. Exit codes: 0 success, 1 verdict=regressed (compare only), 2 usage error.

**Test scenarios.**
1. After a U2 apply + U3 compare cycle, `keep` flips `kept: true` and leaves file untouched
2. After same cycle, `revert` flips `kept: false` AND restores `prompts.py` byte-identical to snapshot
3. Running `keep` or `revert` twice in a row → loud error, no corruption
4. `python -m agent.meta_eval` no args → usage to stderr, exits 2
5. `python -m agent.meta_eval status` prints dashboard including trailing record + recent lessons

---

### U5 — End-to-end manual demo cycle (verification, not code)

**Goal.** Execute the full cycle once with REAL Sonnet output and REAL `prompts.py` edit. This is the deliverable per user constraint ("must produce visible LLM output and real prompts.py changes, not just scaffolding").

**Files.** None — runtime verification only.

**Approach.**
1. **Cycle 1 — `propose`.** `python -m agent.meta_eval propose` → real Sonnet call (~$0.01-0.03). Inspect `data/logs/meta_evals/{ts}.json`. Confirm ≥1 actionable edit with valid anchor + `addresses_pattern_tag` ties it to an observed pattern.
2. **Cycle 1 — `apply`.** `python -m agent.meta_eval apply E1 --yes` → confirm `agent/prompts.py` mutated, snapshot exists, edit_history.jsonl gains pending record.
3. **Cycle 1 — `compare`.** `python -m agent.meta_eval compare "top 3 public safety AI stories"` → ~60-120s, ~$0.05-0.10. Inspect comparison report. edit_history record now has verdict + scores.
4. **Cycle 1 — human verdict.** Read both briefings side-by-side. Decision:
   - Better → `python -m agent.meta_eval keep --note "<why>"`
   - Worse → `python -m agent.meta_eval revert --note "<why>"`
5. **Cycle 2 — verify compounding.** `python -m agent.meta_eval propose` AGAIN. Inspect output's `prior_attempts_considered` field. **VERIFY**: if Cycle 1 was reverted, the same anchor should appear in `prior_attempts_considered` with `decision: skip`. The new proposed edits should NOT repeat the reverted anchor.
6. **(Optional) Cycle 3.** Apply a second edit. Compare. Keep or revert. Confirm `edit_history.jsonl` now has 2 records and the THIRD propose call sees both.

**This is the compounding-loop verification step.** Step 5 is THE test of whether the loop actually compounds vs. just re-proposes the same thing.

**Test scenarios.** Step 5 is the empirical test. If the model re-proposes the same anchor we reverted → meta-eval prompt is not strong enough; iterate on `META_EVAL_PROMPT` wording before declaring U1 done.

**Cost budget.** ~$0.10-0.30 for two full cycles. Within remaining $3-4 API budget.

---

### U6 — `agent/replay.py`: cached-doc summarize replay (canary foundation)

**Goal.** A pure replay layer: re-run ONLY the `summarize` tool on documents already cached in `data/fetched/`, against a swap-in alternate `SYSTEM_PROMPT`. No new search, no new fetch. The mechanism the canary mechanism builds on.

**Files.**
- `agent/replay.py` (new): `replay_summarize(prompt, doc_urls, system_prompt_override) -> dict`
- `tests/test_replay.py` (new)

**Approach.** `replay_summarize()` accepts:
- `prompt`: the original user prompt
- `doc_urls`: list of URLs whose cached fetched docs to use as input
- `system_prompt_override`: optional alternate `SYSTEM_PROMPT` text (None = use current `prompts.SYSTEM_PROMPT`)

Reads cached docs from `storage.load_fetched(url)`. Calls `summarize.run(prompt, documents=fetched_docs)` BUT passes `system_prompt_override` through to the underlying Sonnet call (extends `summarize.run` signature with optional `system` kwarg, default = use module-level prompt). Returns `{briefing: str, tokens: dict, elapsed: float}`. NO state writes — pure function.

**Why a separate module.** Replay is small but distinct: it has no LLM judgment, no file mutations beyond cache reads. Mixing into `meta_eval.py` would force the orchestration tests to depend on cached fetched files. Separation keeps `replay.py` unit-testable in isolation and reusable for offline prompt iteration outside autonomous mode.

**Test scenarios.**
1. Given mocked cached docs + mocked LLM → returns briefing matching the LLM response shape
2. `system_prompt_override=None` → uses `prompts.SYSTEM_PROMPT` (verified via spy on the Anthropic kwargs)
3. `system_prompt_override="alt"` → that exact string reaches the Anthropic call
4. `doc_urls=[]` → returns briefing with summarize's empty-docs stub behavior (no crash)
5. Missing cached doc for one URL → that URL is skipped silently, remaining docs still flow through

---

### U7 — Canary orchestrator + decision logic (the load-bearing autonomy gate)

**Goal.** Given a proposed edit + a baseline `SYSTEM_PROMPT`, run K=3 canary mini-replays and return one of {`promote`, `discard`, `gate`}. THIS is the autonomous mode's safety net.

**Files.**
- `agent/meta_eval.py` (extend): `run_canary(edit_dict, baseline_prompt, k=3) -> dict`, `classify_canary(canary_results) -> str`
- `data/logs/canary/{ts}-E{id}.json` (new, per-canary audit record)

**Approach.** `run_canary()`:
1. Build `candidate_prompt = apply_edit_in_memory(edit, baseline_prompt)` — pure-string transformation, no file I/O
2. Pick K canary prompts: the K most recent COMPLETE runs from `runs.jsonl` with `len(tool_calls.fetch) >= 2` (canaries need cached docs)
3. For each canary prompt P_i:
   - Get its cached fetched doc URLs from `runs.jsonl[i].tool_calls`
   - Run `replay.replay_summarize(P_i, urls, baseline_prompt)` → `briefing_baseline`
   - Run `replay.replay_summarize(P_i, urls, candidate_prompt)` → `briefing_candidate`
   - Score both via `evaluate.check_structure()`
   - Record `delta_i = candidate_score - baseline_score`
4. Persist all K results to `data/logs/canary/{ts}-E{id}.json`
5. Return raw results to `classify_canary()`

`classify_canary(results)` deterministic rules:
- `discard` if any `delta_i < -1` (severe regression on any single canary) OR if `type=delete` (high-blast-radius always gates regardless of canary)
- `promote` if all `delta_i >= 0` AND at least one `delta_i > 0` (strict improvement)
- `gate` otherwise (mixed: some up, some down within ±1 — human judgment needed)

**Critical override for `type=delete`.** Deletions of guidance from `SYSTEM_PROMPT` are never auto-promoted regardless of canary outcome. The blast radius is structural, not visible in 3 summarize runs.

**Cost per canary.** K=3 × 2 summarize calls (baseline + candidate) × ~$0.01 = ~$0.06 per proposed edit. Within budget.

**Test scenarios.**
1. All canaries return delta=+1 → `promote`
2. One canary returns delta=-2, others delta=+1 → `discard` (severe regression rule)
3. One canary delta=+1, one delta=0, one delta=-1 → `gate` (contradictory)
4. `type=delete` with all canaries delta=+1 → `discard` (override rule)
5. Persistence: canary audit file written with all K deltas + the decision

**Why riskiest of the new units.** If `classify_canary` is too permissive, bad edits slip through. If too strict, no edits ever auto-promote and the demo collapses. The thresholds (-1 severe regression, ≥0 floor, strict improvement) are tuneable; empirical validation in U9 is the test of whether these defaults work.

---

### U8 — `autonomous` CLI verb: orchestrator with safety nets

**Goal.** Wire everything into one command: `python -m agent.meta_eval autonomous --prompt "<substring>" [--max-cycles 5] [--max-cost 0.50]`. Produces a single audit-trail markdown when done.

**Files.**
- `agent/meta_eval.py` (extend): `autonomous()`, `_check_safety_nets()`, `_render_autonomous_audit()`
- `data/logs/autonomous_runs/{ts}.md` (new, audit trail)

**Approach.** `autonomous()` per-cycle loop:
1. **Pre-cycle safety check**: total_cost > max_cost → halt. cycle_count > max_cycles → halt. consecutive_regressions >= 2 → halt.
2. **`propose`** (reuses U1 code, picks top-priority edit by `addresses_pattern_tag.count`)
3. **Anchor reproposal guard**: if proposed edit's anchor appears in `edit_history.regressed[*]` → halt + surface ("meta-eval proposed a previously-regressed anchor; META_EVAL_PROMPT needs tightening")
4. **Canary** (U7): `run_canary(edit, current_prompts.SYSTEM_PROMPT)` → decision
5. **Act on decision**:
   - `promote` → snapshot + apply (reuses U2) + record to edit_history with verdict=`canary_promoted` + auto-`keep`
   - `discard` → record skipped attempt to edit_history with verdict=`canary_discarded` (no file mutation, no snapshot)
   - `gate` → halt loop + surface canary results, exit cleanly. Human can resume with manual verbs.
6. **`compare`** (U3, optional in autonomous mode — canary already gave signal; full run gives confirmation but costs more). Skipped by default in autonomous; enable with `--full-compare`.
7. **Update cumulative state**: cost, cycle_count, consecutive_regressions (decrement on promote, increment on canary-discard)
8. **Loop**.

After loop exits, `_render_autonomous_audit()` writes a markdown report: cycles, per-cycle decisions + canary scores + cost, final state of edit_history, current `SYSTEM_PROMPT` diff vs session-start snapshot, halt reason.

**Test scenarios.**
1. With mocked propose returning 1 edit + mocked canary returning `promote` → 1 cycle runs, prompts.py mutated, edit_history record created with verdict=`canary_promoted`, audit log written
2. Mocked canary returns `gate` → loop halts after 1 cycle, halt_reason=`canary_gate`, prompts.py UNCHANGED
3. Mocked canary returns `discard` twice → after second cycle, consecutive_regressions trips halt
4. Cost cap test: with `--max-cost 0.01`, first cycle's canary already exceeds → halt before apply
5. Anchor reproposal: mocked propose returns edit with anchor matching `edit_history.regressed[0].anchor` → halt + surface

---

### U9 — End-to-end autonomous demo cycle (verification, not code)

**Goal.** Real autonomous session. This is the interview demo deliverable.

**Files.** None — runtime verification.

**Approach.**
1. **Prep**: ensure ≥1 eval record exists in `evals.jsonl` (we have 5 runs already). Ensure cached fetched docs exist for ≥3 historical prompts (verified: all 5 runs have cached fetches in `data/fetched/`).
2. **Run autonomous**:
   ```
   python -m agent.meta_eval autonomous \
     --prompt "top 3 public safety AI stories" \
     --max-cycles 3 \
     --max-cost 0.30
   ```
3. **Watch**: stderr streams per-cycle status. Each cycle: `propose → canary K=3 → decision → act`. Expected ~$0.10-0.20 total for 1-3 cycles.
4. **Inspect** `data/logs/autonomous_runs/{ts}.md`. Verify:
   - Halt reason is one of: `max_cycles_hit`, `convergence_no_new_edits`, `canary_gate`, `consecutive_regressions`, `cost_cap`
   - At least 1 promote OR 1 discard recorded (proves canary fired)
   - If promote: `agent/prompts.py` is mutated AND `prompts_history/` has the snapshot
   - `edit_history.jsonl` has new records with `verdict ∈ {canary_promoted, canary_discarded}`
5. **Inspect** `prompts.py` diff vs session-start snapshot. Read the actual evolved `SYSTEM_PROMPT`. Is the change defensible to an interviewer?
6. **Compounding test**: re-run `propose` standalone after autonomous session. Verify `prior_attempts_considered` references at least one of the canary-discarded edits with `decision: skip`. **This is the test of whether the loop compounds across cycles.**
7. **Optional: revert**. If the evolved prompt is worse: `python -m agent.meta_eval revert` restores from last snapshot. Document the failure for future meta-eval improvement.

**This is the demo.** The deliverable is an autonomous session log + an evolved `prompts.py` + a demonstrated compounding test.

**Test scenarios.** N/A — this IS the test.

**Cost budget.** ~$0.10-0.30 for one autonomous session. Within remaining $3-4 API budget.

---

## Deferred (out of scope this cycle)

- Auto-promotion of lessons → durable rules (the three-layer architecture from the timeout: static base + learned rules + rolling lessons). Foundation lives in the eval+edit_history records this plan produces; promotion logic deferred.
- Editing `SUMMARIZE_PROMPT` (separate surface; this cycle only `SYSTEM_PROMPT`).
- Statistical significance of K=3 canary. Production version would use K=10+ with hypothesis testing. Documented in non-goals.
- Cross-session learning of canary calibration (was K=3 enough? Did `discard` correlate with eventual human-revert?). Requires post-hoc human labels.
- Per-prompt-type lesson scoping (current design: lessons are global across all prompt types).
- Streaming the autonomous loop output to a live dashboard (current: file-based audit, read after).
- Auto-running multi-prompt comparison (run 4 historical prompts as broader signal). Manual one-at-a-time per cycle for now.

## Risks

- **R1 — Sonnet proposes vague edits.** Mitigation: U1's `META_EVAL_PROMPT` demands `anchor` field that must match exact substring in current `SYSTEM_PROMPT`; U2 fails loudly if not unique. Garbage in → fail-fast, not garbage out.
- **R2 — One-run comparison is noisy.** Two real runs of the same prompt vary by ±1 structure point from LLM stochasticity. Mitigation: U3 surfaces verdict as `inconclusive` when score unchanged; U7 canary uses K=3 to reduce noise floor.
- **R3 — Anchor drift on repeated edits.** If edit A modifies the line that edit B's anchor sits on, B fails. Mitigation: one edit per cycle in autonomous mode; U7 anchor reproposal guard prevents repeats.
- **R4 — `prompts.py` edit breaks Python syntax.** Mitigation: edits only mutate string-literal content inside `SYSTEM_PROMPT`. Anchor-based `find` is structural. U2 has a post-edit syntax-check via `compile()` before commit.
- **R5 — Canary false positive: K=3 passes but real-world fails.** K=3 is heuristic, not statistical proof. Mitigation: documented in non-goals; U9 includes manual inspection of `prompts.py` diff post-session; revert path one command away.
- **R6 — Canary reward-hacks the structure score.** Edit makes briefings more rule-compliant but emptier. Mitigation: structure-score is necessary-not-sufficient. U9 mandates human read of the actual briefing content as part of the demo verification.
- **R7 — Autonomous mode runs without an API key.** Halts at first `propose` call with clear error. No partial state corruption (no edit_history record opens until `apply` succeeds).
- **R8 — Cost runaway from infinite loop.** Bounded by `--max-cost` (default $0.50) checked at TOP of every cycle. Worst case: one over-budget cycle before halt.

## Verification gates (before declaring done)

**Foundation (manual path):**
1. All U1-U4 unit tests pass (`pytest tests/test_meta_eval.py`)
2. U5 Cycle 1 produces a real `prompts.py` mutation + snapshot + edit_history record + comparison report
3. **U5 Cycle 2 — the manual-path compounding test**: re-`propose` after revert references the prior edit in `prior_attempts_considered` with `decision: skip`

**Hybrid autonomy:**
4. All U6-U8 unit tests pass (`pytest tests/test_replay.py tests/test_meta_eval.py`)
5. U9 autonomous session runs without crash. Halt reason is one of the documented values (not "exception").
6. **U9 canary fired test**: at least one `data/logs/canary/*.json` exists with K=3 results recorded.
7. **U9 decision fired test**: at least one promote OR discard recorded in `edit_history.jsonl` with verdict ∈ {`canary_promoted`, `canary_discarded`}.
8. If promote occurred: `prompts.py` diff vs session-start snapshot is non-empty AND `compile(open('agent/prompts.py').read())` succeeds (no syntax break).
9. **U9 cross-cycle compounding test**: re-`propose` standalone after autonomous session references a prior canary-discarded edit in `prior_attempts_considered` with `decision: skip`.

**Regression:**
10. Existing test suite still passes (`pytest -q` → 61/61 + new replay + new meta-eval tests).

## Next step

Plan at `docs/plans/2026-05-27-001-feat-meta-eval-loop-plan.md`. Moving to execute.

Recommended execution order (riskiest first):
- **U7** (canary classifier) and **U6** (replay) — load-bearing for autonomy. Build + test these BEFORE any autonomous demo. ~30 min combined.
- **U1** (propose with three-input compounding) — load-bearing for everything. ~20 min.
- **U2 + U3** (apply with edit_history + compare with verdict) — manual-path infrastructure also used inside autonomous loop. ~15 min combined.
- **U4** (CLI verbs) + **U8** (autonomous orchestrator) — wiring. ~15 min combined.
- **U5** (manual demo cycle) + **U9** (autonomous demo cycle) — verification. ~10 min runtime each.

**Total estimated build time:** ~90 minutes of code + tests. **Total demo time:** ~15 minutes runtime. **Total cost:** ~$0.30-0.60 of remaining $3-4 API budget.

For interview Part 2 timing: U6+U7+U8+U9 (the autonomous path) is the demo. Manual path (U1-U5) is the fallback that can be invoked individually if anything in autonomous mode misbehaves live.
