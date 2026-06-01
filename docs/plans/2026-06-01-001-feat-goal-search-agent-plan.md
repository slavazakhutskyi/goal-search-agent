---
title: Goal-search agent — universal goal-completion refactor
date: 2026-06-01
status: active
type: feat
origin: docs/brainstorms/2026-06-01-001-goal-search-agent-requirements.md
parent_pr: https://github.com/slavazakhutskyi/rapidsos-intel-agent/pull/1
---

# feat: Goal-search agent (universal goal-completion refactor)

## Summary

Refactor existing intel agent from a single-domain (RapidSOS competitive briefing) into a **universal goal-completion search engine**. The prompt alone selects what's being searched and when to stop. Three acceptance examples must all work end-to-end on a single `python -m agent "<prompt>"` command without per-prompt flags:

- AE1 — freelance leads (companies-as-candidates)
- AE2 — football pitches in Berlin (places-as-candidates, radical domain shift)
- AE3 — competitive intelligence (backward-compat briefing output, preserved unchanged)

The existing meta-eval loop (canary replay, composite verdict, demonstrations) is preserved. The canary's metric switches per-run based on detected output shape: legacy `composite-v1` for briefing runs (AE3), new `goal-v1` for candidates-list runs (AE1, AE2).

## Problem frame (carried from origin)

Today's goal-directed research is manual: Google + Claude chat, 30-90 min per goal, no record of why candidates were kept/dropped, no compounding across goals. Existing agent automates only the briefing step — given named targets, produce a structured briefing. Goal-from-criteria discovery and goal-completion judgment sit upstream and are missing.

(see origin: `docs/brainstorms/2026-06-01-001-goal-search-agent-requirements.md`)

---

## Requirements trace

| Origin ID | Requirement | Covered by |
|---|---|---|
| **A1** Goal-author | Writes single natural-language goal prompt | U5 (SYSTEM_PROMPT design) |
| **A2** Agent | New tools augment search/fetch/summarize; decides when goal met | U1, U2, U3 |
| **A3** Meta-eval | Preserved, adapted metric | U4 |
| **F1** Single-prompt goal completion | search → fetch → score → finalize loop | U1, U2 |
| **F2** Backward-compat briefing path | summarize-terminal path preserved | U3 |
| **F3** Meta-eval over goal runs | canary dispatches metric by output shape | U4 |
| **AE1** Freelance leads | end-to-end candidates_list output ≥10 items ≥0.7 score | U7 verification |
| **AE2** Football pitches Berlin | end-to-end candidates_list output radical domain | U7 verification |
| **AE3** Competitive intelligence | end-to-end briefing_markdown output (backward-compat) | U7 verification |
| **D1** Deterministic + LLM-judge backup | terminator | U2 |
| **D2** Universal score + optional axes | candidate schema | U1 |
| **D3** Output-shape detection per run | shape dispatch | U3 |
| **D4** Repo branding cleanup | README + prompts.py | U5, U6 |
| **S1-S5** Success criteria | All units + verification | U7 |

---

## Key technical decisions

- **KD1 — Single `add_candidate` tool, score required.** Origin OQ3 resolved during planning. Folding `score_candidate` into `add_candidate` (one call, score is required field, optional `axes: dict[str, float]`) reduces tool surface from 3 to 2 (`add_candidate`, `finalize`) and avoids two-call coordination per candidate. Re-scoring an already-added candidate is rare; if needed, the model calls `add_candidate` again with the same URL — implementation idempotency upserts.

- **KD2 — Terminator: deterministic primary, LLM-judge backup.** Loop-side check after each iteration: if `count(candidates where score >= COMPLETION_SCORE_THRESHOLD) >= goal_n`, set `goal_met=True`. `goal_n` parsed from prompt via simple regex (`r"\b(\d+)\b.{0,30}(candidates?|companies?|places?|leads?|results?|items?|options?)"`) with default 10 if not found. When deterministic check cannot evaluate (regex fails AND no default applies — e.g., "find a good restaurant"), invoke a single Haiku judge call per iteration: "Given the goal and the candidate list so far, is this enough? yes/no + reason." Conservative: judge defaults `no` on ambiguity.

- **KD3 — Output-shape detection: terminal-tool dispatch in `loop.py`.** Origin OQ1 resolved. After loop exit, agent's runtime inspects the last successful tool call: `finalize` → emit `candidates_list` markdown built from accumulated candidates; `summarize` → emit briefing (existing path); neither → existing fallback. No CLI flag, no config — promp shape dictates output shape.

- **KD4 — Goal-coverage metric reuses operator-sim shape.** New `agent/operator_sim.py:goal_coverage_score(prompt, candidates_list)` mirrors `coverage_score(prompt, briefing)`. One Haiku call, returns coverage∈[0,1]. Composite verdict in canary dispatches by output_shape detected on baseline/candidate replays. New `metric_version: "goal-v1"` tags goal-output records; existing `composite-v1` continues for briefing-output records.

- **KD5 — Candidates accumulator in loop state.** Candidates live in `loop.run`'s local state (list of dicts), not in storage. Each iteration, model can `add_candidate` to grow the list. Final candidates_list rendered to markdown at loop exit. No persistence between runs (matches origin Out-of-scope: persistent multi-session candidate stores).

- **KD6 — Backward-compat preserved as-is.** Existing `summarize` tool, `SUMMARIZE_PROMPT`, briefing structure check, and composite verdict remain functional. AE3 path runs through them unchanged. Only the prompt rewording is unavoidable (U5) to add candidates-flow guidance; the new SYSTEM_PROMPT must explicitly state that briefing flows are also supported via `summarize` terminal call.

---

## High-Level Technical Design

*Directional guidance, not implementation specification.*

```
loop.run(prompt)
  ├─ parse_goal_metadata(prompt) → {goal_n: int, mode_hint: "candidates" | "briefing" | None}
  │
  └─ for iteration in MAX_ITERATIONS:
       └─ llm_call(SYSTEM_PROMPT + tools)
            ├─ tool_use: search        → existing dispatch
            ├─ tool_use: fetch         → existing dispatch
            ├─ tool_use: summarize     → existing dispatch + last_summarize_briefing
            ├─ tool_use: add_candidate → loop_state.candidates.append(or upsert)
            ├─ tool_use: finalize      → set finalized=True
            └─ text (no tool)          → terminate

       └─ post-tool check (KD2 terminator):
            ├─ if finalized: break
            ├─ elif det_check(candidates, goal_n): inject "goal met, call finalize" hint
            └─ elif iteration >= MIN_FOR_JUDGE: judge_call → if yes: hint

  └─ post-loop dispatch (KD3):
       ├─ if finalized OR len(candidates) >= 3: render candidates_list markdown
       ├─ elif last_summarize_briefing: emit briefing (existing)
       └─ else: existing fallback
```

```
canary verdict dispatch (KD4):
  baseline_run, candidate_run = replay_run(canary_prompt, both prompts)
  shape = detect_shape(baseline_run.briefing)   # heuristic: candidates header vs TL;DR
  if shape == "candidates":
      use goal_coverage_score → composite breakdown: {goal_coverage, efficiency, errors}
      metric_version = "goal-v1"
  else:
      use coverage_score + check_structure → existing composite
      metric_version = "composite-v1"
```

---

## Implementation Units

### U1. New tools: `add_candidate` + `finalize`

**Goal.** Two new tools the model can call: `add_candidate(url, name, why_fits, score, axes?)` accumulates candidates in loop state; `finalize()` signals "goal met, emit candidates list."

**Requirements.** A2, F1, D2 (score in [0,1] + optional axes), KD1, KD5.

**Dependencies.** None (foundational unit).

**Files.**
- `agent/tools/candidates.py` (new) — `add_candidate.TOOL_SCHEMA`, `add_candidate.run()`, `finalize.TOOL_SCHEMA`, `finalize.run()`. Both stateless; mutation handled by caller (loop holds state).
- `agent/loop.py` (modify) — register both tools in `TOOL_REGISTRY`; add loop-local `candidates: list[dict]` state; add dispatch for `add_candidate` (upsert by URL) and `finalize` (set flag).
- `tests/test_candidates.py` (new) — schema shape, idempotent upsert, score validation.
- `tests/test_loop.py` (modify) — integration scenario: model emits add_candidate × 3 + finalize → loop state has 3 candidates and `finalized=True`.

**Approach.** Tools follow the existing `search.py` / `fetch.py` pattern — `TOOL_SCHEMA` dict + `run(**kwargs)` function returning a dict. `add_candidate.run` returns `{"ok": true, "candidate": {...}}` for the model's tool_result. `finalize.run` returns `{"finalized": true}`. The actual upsert/state mutation happens in `loop.py`'s dispatch wrapper because the candidates list is loop-local — the tool itself is a pure-validation pass-through. This keeps `agent/tools/candidates.py` testable in isolation.

**Patterns to follow.** `agent/tools/search.py` (TOOL_SCHEMA shape, graceful error dict). `agent/loop.py:_dispatch_tool` and the existing summarize auto-attach hook (state-aware dispatch).

**Test scenarios.**
- Happy path: `add_candidate.run(url="https://a.example", name="A", why_fits="...", score=0.8)` returns `{"ok": true, "candidate": {...}}`.
- Score validation: `score=1.5` returns `{"error": "score must be in [0, 1]"}`.
- Missing required: `add_candidate.run(url="x")` (no name) returns error dict.
- Optional axes: `add_candidate.run(..., axes={"timezone": 0.9, "skill": 0.7})` accepts and round-trips.
- `finalize.run()` returns `{"finalized": true}`.
- Integration (loop): model emits 3× `add_candidate` then `finalize` → `loop.run` result includes `candidates: [...]` length 3 and `finalized: True`.
- Integration (upsert): model emits `add_candidate(url=X, score=0.5)` then `add_candidate(url=X, score=0.7)` → state has 1 candidate with score 0.7 (upsert, not duplicate).

**Verification.** New tool schemas appear in the model-visible tool list when `loop.run` is invoked. A canned mock conversation with 3 add_candidate + finalize sequences results in loop state containing exactly 3 candidate dicts.

---

### U2. Goal-completion terminator (deterministic + LLM-judge backup)

**Goal.** After each iteration, check whether the goal is met. Inject a structured hint into the next assistant turn if so. Halt the loop cleanly on `finalize` OR cap hit.

**Requirements.** F1, D1, KD2.

**Dependencies.** U1.

**Files.**
- `agent/loop.py` (modify) — add `_parse_goal_n(prompt) -> int`, add deterministic check after dispatch, add LLM-judge call gated by `MIN_FOR_JUDGE` iteration. Both inject a synthetic user message: `"You have N candidates with score ≥ T. Call finalize() now to emit the list."`
- `agent/llm.py` (no change expected; uses existing `call_with_retry`).
- `tests/test_loop.py` (modify) — terminator scenarios.

**Approach.** Deterministic check is pure-function on `loop_state.candidates`: count entries with `score >= COMPLETION_SCORE_THRESHOLD` (constant in `loop.py`, default 0.7). Compare to `goal_n` parsed from prompt via regex (KD2 spec). If `count >= goal_n`, the loop appends a hint message in the next iteration. Independently, `finalized=True` from U1 ends the loop immediately.

LLM-judge backup: fires only when `_parse_goal_n` returns None AND iteration ≥ `MIN_FOR_JUDGE` (default 3, so the model has had time to add a few candidates first). Single Haiku call with prompt: `"Goal: {prompt}\nCandidates so far ({n}): {one-line each}\nIs this enough to satisfy the goal? Answer JSON: {decision: yes|no, reason: ...}"`. Cached per (prompt, len(candidates)) tuple to avoid repeated calls.

**Patterns to follow.** Existing `MAX_ITERATIONS` cap + early-break pattern in `loop.py`. Existing `llm.call_with_retry` invocation in `agent/operator_sim.py:coverage_score` (response parsing, JSON cleanup, fallback to `no` on parse failure).

**Test scenarios.**
- `_parse_goal_n("find 15 leads for X")` returns 15.
- `_parse_goal_n("find a good restaurant")` returns None (no number).
- `_parse_goal_n("competitive intelligence on X, Y, Z")` returns None (no number near candidate-noun).
- Deterministic terminator: 10 candidates with score ≥ 0.7 + goal_n=10 → hint injected.
- Deterministic terminator: 10 candidates with score 0.5 (below threshold) + goal_n=10 → no hint.
- `finalize` from U1 → loop breaks at next iteration regardless of count.
- LLM-judge: no goal_n parsed + iteration=3 + judge returns `yes` → hint injected.
- LLM-judge cache: same (prompt, len(candidates)) called twice → only one LLM call (spy).
- LLM-judge defaults `no` on malformed JSON.

**Verification.** Mocked model conversation reaches goal_n=3 → next iteration's user message contains the synthetic hint. Same conversation without reaching threshold → no hint injected. Test asserts both states.

---

### U3. Output-shape detection + dispatch

**Goal.** After loop exit, emit one of two output shapes based on which terminal tool fired. Preserves AE3 backward-compat unchanged; produces a clean candidates_list markdown for AE1/AE2.

**Requirements.** F1, F2, AE3 (backward-compat), D3, KD3, KD6.

**Dependencies.** U1, U2.

**Files.**
- `agent/loop.py` (modify) — post-loop dispatch logic. New helper `_render_candidates_list(prompt, candidates)` producing markdown.
- `agent/prompts.py` (modify in U5; this unit only adds helper).
- `tests/test_loop.py` (modify) — shape-dispatch scenarios.

**Approach.** After the iteration loop exits:
1. If `finalized=True` OR `len(candidates) >= 3`: render candidates_list. Sort by score desc. Markdown shape:
   ```
   # Goal — {prompt[:80]}
   *N candidates · sorted by score desc*

   ## Candidates
   1. **{name}** ({url}) — score {score:.2f}
      {why_fits}
      {axes summary if present}
   ...
   ```
2. Else if `last_summarize_briefing`: emit briefing (existing path).
3. Else: existing fallback (last text, partial_empty_response, etc.).

The `len(candidates) >= 3` floor exists so a partial run that fetched candidates but never called `finalize` still emits a useful list rather than collapsing to fallback. AE3 path never triggers it (zero `add_candidate` calls).

**Patterns to follow.** Existing post-loop dispatch in `loop.run` (the `last_summarize_briefing` selection logic + partial fallbacks at lines ~145-170 of `agent/loop.py`).

**Test scenarios.**
- `finalized=True` + 5 candidates → candidates_list output, sorted by score desc.
- `finalized=False` + 4 candidates (no finalize) → candidates_list output (floor).
- `finalized=False` + 2 candidates → fallback to briefing or text (below floor).
- `finalized=False` + 0 candidates + `last_summarize_briefing` set → briefing output (AE3 path).
- Markdown shape: rendered output contains `# Goal —`, `## Candidates`, numbered list with `**name**`, URL, score, why_fits.
- AE3 regression: a run that calls only search + fetch + summarize produces a briefing identical in shape to the existing post-PR#1 output.

**Verification.** Two parallel test runs through `loop.run` with the same mocked tools: one with `add_candidate × 3 + finalize` produces a candidates_list briefing; the other with `search + fetch + summarize` produces the existing-format briefing. Both pass without mode flags.

---

### U4. Goal-coverage metric + composite verdict dispatch

**Goal.** New `goal_coverage_score(prompt, candidates_list_markdown)` in `agent/operator_sim.py`. New `metric_version="goal-v1"` tag. Canary `run_canary` dispatches metric by detected output shape.

**Requirements.** A3, F3, KD4, S3.

**Dependencies.** U3.

**Files.**
- `agent/operator_sim.py` (modify) — add `goal_coverage_score(prompt, candidates_list_markdown)` mirroring `coverage_score`. Add helper `_detect_output_shape(text) -> Literal["candidates", "briefing", "unknown"]`.
- `agent/meta_eval.py` (modify) — in `run_canary`, after replay runs return, call `_detect_output_shape` on each. Build `eval_record` for composite_score using the matching scorer (existing structure+coverage for briefing; new goal_coverage_score + structural-lite for candidates). Set top-level `metric_version` accordingly.
- `agent/evaluate.py` (modify) — export `GOAL_METRIC_VERSION = "goal-v1"` alongside `COMPOSITE_METRIC_VERSION`. Update `load_few_shot_examples` to accept records with EITHER version (gate widens to `metric_version in (COMPOSITE_METRIC_VERSION, GOAL_METRIC_VERSION)`).
- `tests/test_operator_sim.py` (modify) — goal_coverage_score tests + output-shape detection tests.
- `tests/test_meta_eval.py` (modify) — canary verdict dispatch tests (mocked replay returning each shape).

**Approach.** `goal_coverage_score` follows the same prompt structure as `coverage_score`: K=5 judge probes asking "does this candidates list cover the goal? are any candidates obviously low-quality? does the count match the requested N?". Returns coverage∈[0,1] + breakdown. Output-shape detection is a 5-line heuristic: candidates markdown contains `## Candidates` header AND numbered items; briefing markdown contains `## TL;DR` and `## Sources`. Default `unknown` if neither.

`run_canary` per-K-iteration logic switches scorer based on detected shape on the BASELINE briefing (candidate run uses the same scorer, even if its detected shape differs — comparison must be apples-to-apples). Top-level `metric_version` set to whichever scorer fired.

**Patterns to follow.** Existing `coverage_score` shape and caching in `agent/operator_sim.py`. Existing `composite_score` arithmetic and breakdown structure. Existing `run_canary` K-loop in `agent/meta_eval.py:run_canary`.

**Test scenarios.**
- `goal_coverage_score(prompt, candidates_markdown)` with mocked Haiku returning grounded YES/NO → coverage matches expected ratio (analog to existing `coverage_score` tests).
- `_detect_output_shape("# Goal — X\n## Candidates\n1. **a** ...")` returns `"candidates"`.
- `_detect_output_shape("# Briefing — X\n## TL;DR\n...\n## Sources\n[^1]...")` returns `"briefing"`.
- `_detect_output_shape("random text")` returns `"unknown"`.
- `run_canary` with mocked replay returning briefing both sides → top-level `metric_version == "composite-v1"`, existing breakdown.
- `run_canary` with mocked replay returning candidates_list both sides → top-level `metric_version == "goal-v1"`, breakdown contains `goal_coverage` instead of `structure`.
- `run_canary` with mismatched shapes (baseline=briefing, candidate=candidates) → uses baseline's scorer for both (apples-to-apples), logs the mismatch in canary record.
- `load_few_shot_examples` accepts records with `metric_version="goal-v1"` alongside `composite-v1` (no exclusion).

**Verification.** Canary records produced by `run_canary` after a goal-output replay contain `metric_version: "goal-v1"` and a `goal_coverage` field in the breakdown. Canary records from a briefing-output replay continue to produce `composite-v1` with the existing breakdown.

---

### U5. SYSTEM_PROMPT rewrite — generic goal-search + tools guidance

**Goal.** Replace RapidSOS-specific SYSTEM_PROMPT with generic goal-completion prompt that documents the new tools and both output shapes (candidates_list, briefing).

**Requirements.** A1, A2, D4 (branding cleanup), KD6.

**Dependencies.** U1 (to know tools), U3 (to know output shapes).

**Files.**
- `agent/prompts.py` (modify) — rewrite `SYSTEM_PROMPT` from scratch. Keep `SUMMARIZE_PROMPT` unchanged (still used by AE3 briefing path).
- `tests/test_loop.py` (modify) — smoke test that SYSTEM_PROMPT contains the new tool names + both flow descriptions.

**Approach.** New SYSTEM_PROMPT covers:
- Identity: "goal-completion search agent. Single user prompt = single goal. Decide when goal is met."
- Tools: `search`, `fetch`, `summarize` (existing) + `add_candidate`, `finalize` (new).
- Strategy:
  1. Read the goal. Parse implicit goal-N if present.
  2. Search 1-3 focused queries.
  3. Fetch promising URLs.
  4. **For each promising candidate**: call `add_candidate` with score [0,1] and a 1-2 sentence `why_fits` grounded in fetched content.
  5. When goal-N reached: call `finalize`. Loop emits ranked list.
  6. **Alternative briefing flow** (when prompt asks for narrative analysis on named targets, not a count of candidates): call `summarize` instead of `finalize`. Existing briefing path runs.
- Stop conditions stated explicitly: finalize OR summarize OR no further tools.
- Removed: all RapidSOS-specific text (competitor names, parent-company mapping, intel-domain language).
- Preserved: tool-use discipline (do not narrate reasoning between tool calls, prefer recent sources for time-bounded goals).

**Patterns to follow.** Existing SYSTEM_PROMPT structure in `agent/prompts.py` (identity → tools → strategy → constraints).

**Test scenarios.**
- SYSTEM_PROMPT contains the strings `add_candidate`, `finalize`, `summarize`. Smoke test.
- SYSTEM_PROMPT does NOT contain `RapidSOS`, `Carbyne`, `Axon`, `RapidDeploy`, `Prepared`. Branding-cleanup test.
- SYSTEM_PROMPT does NOT contain `parent company` / `parent companies` rule.
- Prompt length under model context budget (assert `< 4000 chars` as soft cap).

**Verification.** A live `python -m agent` invocation with a goal-N prompt shows the model calling `add_candidate` (visible in tool_calls log). A live invocation with a named-targets briefing prompt shows the model calling `summarize` directly (AE3 backward-compat path active).

---

### U6. README + branding cleanup

**Goal.** Repo presents as generic goal-search agent. AE1+AE2+AE3 included as worked examples. Zero RapidSOS-specific strings outside `docs/solutions/` and `docs/plans/` (which preserve historical context).

**Requirements.** D4, S4.

**Dependencies.** U5 (so README docs match new prompt).

**Files.**
- `README.md` (rewrite) — new generic framing, AE1/AE2/AE3 examples, brief architecture overview pointing at `docs/plans/` and `docs/solutions/`, install + quickstart.
- `agent/__main__.py` (modify) — CLI usage string updated.
- `agent/evaluate.py` (review) — the `JUDGE_PROMPT` references "RapidSOS operator" — generalize to "the user who wrote this prompt" (the operator role becomes the prompt-author).
- `agent/operator_sim.py` (review) — `OPERATOR_SIM_PROMPT` similarly. Replace "RapidSOS operator who has 3 minutes to triage" with "the user who wrote this prompt and has 3 minutes to triage."
- `docs/solutions/*` (no change — historical preservation).
- `docs/plans/*` (no change — historical preservation).

**Approach.** README structure:
1. Title + 1-paragraph elevator pitch.
2. Three worked examples (AE1 + AE2 + AE3), each with the literal command + expected output shape.
3. Quickstart: clone, `pip install`, set `ANTHROPIC_API_KEY`, run.
4. Architecture: 3-bullet overview of the loop + meta-eval cycle. Link `docs/plans/` and `docs/solutions/` for depth.
5. Status: this is a portfolio piece, not production.

Branding cleanup is mechanical: grep for RapidSOS/Carbyne/RapidDeploy/Prepared/Axon in non-historical files; replace or remove. Audit reports both `README.md` + `agent/` Python files.

**Patterns to follow.** Existing README structure (well-formed). Existing JUDGE_PROMPT shape in `agent/evaluate.py:JUDGE_PROMPT`.

**Test scenarios.**
- Grep test: `git ls-files agent/ README.md | xargs grep -lE "RapidSOS|Carbyne|RapidDeploy|Prepared|Axon" | grep -v docs/` returns empty. (Negative grep — could be a CI check or a Python test that runs the grep via subprocess.)
- JUDGE_PROMPT contains "user" or generic phrasing, not "RapidSOS operator".
- README contains the three example commands verbatim.
- CLI usage string from `python -m agent` (no args) does not mention RapidSOS.

**Verification.** Manual: read the README as a stranger to the repo. Does it explain what the tool does and how to run it in under 90 seconds? Run all three example commands; all three succeed.

---

### U7. End-to-end live verification (AE1 + AE2 + AE3)

**Goal.** Prove all three acceptance examples work end-to-end under live LLM calls. Capture cost per AE for the verification log.

**Requirements.** AE1, AE2, AE3, S1, S2, S5.

**Dependencies.** U1, U2, U3, U4, U5, U6.

**Files.** None new. Runtime verification only. Output captured in commit message + a brief `docs/solutions/2026-06-XX-goal-search-verification.md` summary if useful.

**Approach.**
1. Confirm `pytest -q` passes — all units green, no regression from PR#1's 140 tests.
2. Run AE3 first (lowest risk, backward-compat). Verify output matches existing briefing shape, cost ≤ $0.30.
3. Run AE1 (companies-as-candidates). Verify output is candidates_list ≥10 entries with scores, sorted desc. Cost ≤ $0.30.
4. Run AE2 (places-as-candidates). Verify output is candidates_list ≥10 entries, no code change from AE1 (proves universality). Cost ≤ $0.30.
5. Inspect `data/logs/runs.jsonl` for new entries. Verify `data/logs/evals.jsonl` records new eval per run with appropriate `metric_version`.
6. Run one autonomous meta-eval cycle on the goal-shape output. Verify canary record carries `metric_version: "goal-v1"`.

**Patterns to follow.** Existing verification pattern from PR#1 (the "live verification" commits b24203c).

**Test scenarios.** None (runtime verification, not unit tests).

**Verification.** Three live runs produce three valid outputs of the expected shapes. `data/logs/evals.jsonl` carries one record per run, with shape-correct `metric_version` tag. Total verification cost ≤ $1.00 of the ~$1.50 remaining budget.

---

## System-wide impact

- **Agent loop (`agent/loop.py`)** — most changes concentrate here: new dispatch state, terminator hooks, post-loop shape dispatch. Risk: regression of AE3 path. Mitigation: U3 test scenario explicitly asserts AE3-shaped output is byte-comparable to a pre-change baseline.

- **Meta-eval (`agent/meta_eval.py`, `agent/operator_sim.py`)** — dual-metric dispatch. Risk: `load_few_shot_examples` regression if the gate gets the version filter wrong. Mitigation: explicit test that both `composite-v1` and `goal-v1` records flow through.

- **Persistence formats** — `data/logs/evals.jsonl`, `data/logs/edit_history.jsonl`, `data/logs/canary/*.json` gain `metric_version: "goal-v1"` records alongside existing `composite-v1` records. No schema migration: defensive `.get()` reads handle absence.

- **CLI contract (`python -m agent "<prompt>"`)** — surface unchanged. Same single positional argument. Same env vars. Output stdout shape changes per prompt (candidates_list vs briefing), but stderr status line format is preserved.

---

## Risks & mitigations

- **R1 — AE3 regression.** Model could route a backward-compat briefing prompt through the candidates path under the new SYSTEM_PROMPT. *Mitigation:* U5 SYSTEM_PROMPT explicitly documents both flows with examples; U3 floor (`>= 3 candidates`) prevents accidental candidates_list output when the model didn't actually add candidates; U7 runs AE3 first as the lowest-risk smoke test.

- **R2 — Terminator deadlock.** Deterministic check could miss a satisfied goal if regex parses goal-N wrong. *Mitigation:* MAX_ITERATIONS cap unchanged from existing loop (12). LLM-judge backup. Worst case: model hits cap, candidates_list rendered from accumulated entries (U3 floor).

- **R3 — Goal-coverage metric LLM noise.** New `goal_coverage_score` Haiku call may be miscalibrated. *Mitigation:* same K=5 grounded-citations pattern as existing `coverage_score` (defeats LLM-judge generosity bias). Composite verdict has 4 components — operator-coverage is 50%, structural floor and efficiency anchor the rest. If goal-coverage proves noisy in U7, weights are tunable post-ship.

- **R4 — Cost overrun on verification.** Three live AEs + one autonomous cycle could approach $1.50. *Mitigation:* AE3 is cheapest (~$0.10 with backward-compat cache hits). AE1+AE2 each ~$0.20-0.30. Autonomous cycle ~$0.10 with Haiku. Total ~$0.80, within budget. If overrun: defer autonomous cycle verification.

- **R5 — Model confusion between `summarize` and `finalize`.** New tool surface (5 tools total). *Mitigation:* SYSTEM_PROMPT spells out: "use `finalize` when emitting a ranked list of candidates; use `summarize` when emitting a narrative briefing on named targets." Worked example for each in U5 prompt text.

---

## Deferred to follow-up work

- Persistent multi-session candidate stores (origin Out-of-scope).
- Domain-specific scoring rubrics beyond universal score + axes (origin Out-of-scope).
- Hosted service / API / web UI (origin Out-of-scope).
- Multi-provider LLM support (origin Out-of-scope).
- Auto-follow-up goal generation from prior runs.
- Repo rename `rapidsos-intel-agent` → `goal-search-agent` on GitHub (deferred for URL stability of existing PR#1 and external links; code-level branding cleanup covers the portfolio-readability goal).

---

## Verification gates

- All new unit tests pass (`pytest tests/test_candidates.py tests/test_loop.py tests/test_operator_sim.py tests/test_meta_eval.py tests/test_evaluate.py`).
- Existing test suite (140/140 on main) continues to pass — no regression.
- AE1 live run produces candidates_list with ≥10 entries, sorted by score desc. Cost ≤ $0.30.
- AE2 live run produces candidates_list with ≥10 entries (different domain — places). No code change from AE1. Cost ≤ $0.30.
- AE3 live run produces briefing in existing format. Cost ≤ $0.30.
- `data/logs/evals.jsonl` carries `metric_version: "goal-v1"` for AE1+AE2 runs and `composite-v1` for AE3.
- Grep test: no RapidSOS-specific strings in `agent/` or `README.md` outside `docs/` historical preserves.
- One autonomous meta-eval cycle on a goal-shape prompt produces a canary record with `metric_version: "goal-v1"`.

---

## Implementation-time unknowns (deferred to execution)

- Exact `_render_candidates_list` markdown format may evolve based on what reads well in stdout.
- Exact regex for `_parse_goal_n` — may need tuning against AE2 ("10 best amateur football pitches") and similar phrasings.
- Goal-coverage prompt wording — K=5 stays fixed; exact question phrasing may iterate during U7 verification.
- Whether the LLM-judge terminator needs a result cache key beyond `(prompt, len(candidates))` — empirical.
- Whether `add_candidate` upsert-by-URL needs an additional dedup pass on `name` (URL collisions across re-fetches are unlikely but possible).

---

## Build order & estimated cost

Sequential dependency order: U1 → U2 → U3 → U4 → U5 → U6 → U7.

| Unit | Est. wall time (agent speed) | Est. cost |
|---|---|---|
| U1 | 30 min | 0 |
| U2 | 30 min | 0 |
| U3 | 20 min | 0 |
| U4 | 45 min | 0 |
| U5 | 20 min | 0 |
| U6 | 20 min | 0 |
| U7 | 30 min runtime | $0.50-0.80 |
| **Total** | ~3 hours code + 30 min verification | **~$0.50-0.80** |

Within remaining $1.50 budget.

---

## Next step

Recommended: invoke `/ce-work` against this plan immediately. Sequential execution, commit per unit, run tests after each.
