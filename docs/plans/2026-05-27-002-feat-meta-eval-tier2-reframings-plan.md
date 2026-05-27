# Plan — Meta-eval Tier 2 reframings: operator simulation + few-shot compounding + code-level edits

> Created: 2026-05-27 · Depth: Deep · Source: post-eval re-planning session
> Parent plan: `docs/plans/2026-05-27-001-feat-meta-eval-loop-plan.md`
> Eval origin: `docs/solutions/2026-05-27-meta-eval-strict-evaluation.md`
> Mode: hybrid autonomy continues; this plan **replaces the scoring substrate** that hybrid mode uses, and **expands the edit surface** from prompts to code.

## Goal

The strict evaluation revealed three structural limits in the current system:
1. **The canary metric is saturated** — structure-score caps at 5/5, blind to token efficiency and content quality. A canary that should have promoted (50% token reduction, same structure) got gated.
2. **Lessons-as-rules don't transfer** — abstract bullets in SYSTEM_PROMPT do not reliably change behavior. The dominant bug (`summarize_missing_documents`) recurred in 6 of 7 runs *with the lesson active*.
3. **Prompts can't fix what code broke** — the most chronic failure is a tool-schema issue, but the meta-eval can only propose prompt edits.

This plan addresses all three by **reframing** what the system measures, how it teaches itself, and what it can edit. Together, these convert the loop from "demonstrably bounded autonomy" to "system that actually improves measurably."

**Non-goals.** Statistical significance testing (K=10+, hypothesis tests). Production observability (LangFuse, dashboards). Multi-tenant deployment. These are Tier 3 — explicitly deferred per user direction.

---

## Three reframings

### Reframing A — Operator simulation replaces structure-score

**What.** Stop measuring briefing form. Start measuring briefing utility to its actual user.

**How.** A small Haiku call: given the user prompt and the briefing, generate 5 follow-up questions a RapidSOS operator would ask in the first 3 minutes of reading. Then judge each: does the briefing already answer this question?

Composite verdict for the canary:
```
score = w_op * operator_sim_coverage      # 0..1, primary signal
      + w_struct * structure_score / 5    # 0..1, sanity floor
      + w_eff * (-token_delta_normalized) # punish bloat, reward efficiency
      + w_err * (-summarize_error_count)  # surface tool-contract failures
```

Default weights: `w_op=0.5, w_struct=0.2, w_eff=0.2, w_err=0.1`. Tuneable per
deployment. Promote if composite delta > 0.05; discard if < -0.10; gate
otherwise. Thresholds anchored to "noticeable change" rather than "any change."

**Why this matters.** Operator-sim is *unbounded* — a briefing can always answer 1 more question better. The structure-saturation collapse goes away. And the metric directly aligns with what RapidSOS would actually pay for.

---

### Reframing B — Few-shot demonstrations replace abstract lessons

**What.** Stop appending rules. Start appending behavior examples.

**How.** Today the lessons block looks like:

```
## Recent lessons
- Call summarize with BOTH prompt and documents
- Stop searching after 4 URLs
```

After this reframing:

```
## Good runs (compounding from kept edits)
Example: Prompt "...competitive intel..." → 3 searches, 5 fetches, 1 summarize,
score 0.78, briefing answers 4/5 operator questions

## Anti-patterns (from reverted attempts)
Example: Prompt "...competitive intel..." → 8 searches, hit cap, fallback
inline output, score 0.20
```

Demonstrations carry more behavioral signal than rules. Same token budget produces stronger behavior shaping.

**Source.** `edit_history.jsonl` (which runs were kept/reverted) cross-referenced with `runs.jsonl` (tool-call traces) — already persisted. Just needs an extractor + formatter.

---

### Reframing C — Code-level meta-eval

**What.** Allow the meta-eval to propose code changes, not just prompt edits.

**How.** Extend the proposed-edit schema:

```json
{
  "id": "E1",
  "type": "code",
  "file": "agent/tools/summarize.py",
  "anchor_function": "run",
  "diff_unified": "...",
  "test_diff_unified": "...",
  "rationale": "..."
}
```

Canary for code edits runs:
1. Apply diff to a scratch copy
2. Run `pytest -q` against scratch copy → must pass
3. Run agent on cached canary prompt → score via composite verdict
4. If structure breaks OR tests fail → discard
5. Otherwise → **always gate** (code changes never auto-promote; the human reviews)

**Hard rule:** code edits ALWAYS gate for human review regardless of canary outcome. The blast radius is too large for outcome-only autonomy. This makes the autonomous path explicitly bounded for code: meta-eval *proposes*, canary *validates*, human *commits*.

**Why this matters.** The chronic `summarize_missing_documents` bug is one code change away from extinction (make `documents` optional in `summarize.TOOL_SCHEMA` + auto-attach from loop state). Today's system cannot propose this. With C, it can — and the resulting eval data is no longer polluted by a code bug masquerading as a prompt problem.

---

## Architecture choice

Three new modules + extensions to existing ones:

- **`agent/operator_sim.py`** (new, ~80 lines) — operator simulation: question generation + answer judgment + score aggregation
- **`agent/canary_score.py`** (new, ~60 lines) — composite verdict computation. Pure function, easily unit-testable
- **`agent/code_edit.py`** (new, ~120 lines) — code-edit application sandbox: scratch-copy, diff apply, pytest runner, rollback
- **`agent/evaluate.py`** (extend) — `load_few_shot_examples()` reads edit_history + runs.jsonl, formats as demonstrations; replaces `lessons_block()` with `demonstrations_block()`
- **`agent/meta_eval.py`** (extend) — `META_EVAL_PROMPT` updated to allow `type: code` in proposed_edits; `run_canary` accepts code edits with new branch; `classify_canary` consumes composite score; `autonomous` orchestrator gates code edits hard
- **`agent/prompts.py`** (target of edits, unchanged structure)

Why three new modules instead of folding into meta_eval: each has independent purpose, independent test surface, and reusable scope. `operator_sim` should be invokable on any briefing for offline evaluation. `canary_score` is a pure function. `code_edit` is sandboxing infrastructure. Mixing into `meta_eval.py` would push it past 1500 lines and couple unrelated tests.

---

## Pre-work: clear the noise floor

Before any of the reframings, fix the dominant code bug that's polluting eval data. This is **Tier 1b** from the prior list, promoted to a prerequisite here because every Tier 2 reframing operates on data that this bug currently dominates.

### U0 — Make `summarize.run(documents=None)` accept missing arg

**Goal.** Eliminate the `summarize_missing_documents` pattern entirely by making the tool contract permissive.

**Files.**
- `agent/tools/summarize.py` (modify): `documents` default `None`; if `None`, the loop is expected to attach cached docs (see below)
- `agent/loop.py` (modify): when dispatching `summarize` tool call, if input dict lacks `documents`, auto-attach all successfully-fetched docs from the current run's tool history
- `agent/tools/summarize.py` (modify): `TOOL_SCHEMA` — remove `documents` from `required`, update description to say "documents optional; loop will attach all fetched docs if omitted"
- `tests/test_loop.py` (modify): add test that summarize without documents arg auto-attaches and succeeds

**Approach.** The loop already tracks fetched docs in `tool_calls`. When the model calls `summarize(prompt="...")` without `documents`, the dispatch wrapper reads cached docs for every successful fetch this run and injects them. No model behavior change required — the existing prompt strategy still works, but the failure case where the model omits `documents` now succeeds.

**Test scenarios.**
1. Model calls `summarize(prompt="X")` → loop auto-attaches fetched docs → summarize succeeds
2. Model calls `summarize(prompt="X", documents=[...])` → existing behavior unchanged
3. Model calls `summarize(prompt="X")` with zero successful fetches → summarize gets empty list, returns the empty-docs stub briefing (existing behavior)
4. End-to-end: run agent, induce `summarize` without documents via mocked model — should NOT log a trace_issue

**Verification.** Run a full live agent invocation after this lands. The next eval record should show `summarize_missing_documents` is absent from `trace_issues` even if the model "incorrectly" calls summarize without the arg.

---

## Implementation units

### U1 — `agent/operator_sim.py`: operator follow-up simulation

**Goal.** Single Haiku call that turns a (prompt, briefing) pair into a coverage score: "what fraction of the operator's likely follow-up questions does this briefing already answer?"

**Files.**
- `agent/operator_sim.py` (new)
- `tests/test_operator_sim.py` (new)

**Approach.** Two stages, one Haiku call combined via structured prompt:

```
OPERATOR_SIM_PROMPT = """You are a RapidSOS operator who has 3 minutes to triage <briefing>.

User prompt: {prompt}
Briefing: {briefing}

Step 1: List 5 follow-up questions you'd most want answered in your next 30 minutes of work.
Step 2: For each question, answer YES/NO: does the briefing as-written already answer it?

Return STRICT JSON:
{{
  "questions": ["...", "...", ...],
  "answered": [true, false, true, true, false],
  "coverage": 0.6
}}
"""
```

Coverage = `sum(answered) / len(questions)`. Pure compositional — feeds into composite verdict.

Caching: same briefing → same operator-sim score. Cache by `hash(briefing)` to avoid re-paying for repeated evaluations within one autonomous session.

**Test scenarios.**
1. Mocked Haiku response → returns coverage between 0 and 1 with matching question/answer length
2. Briefing answering 5/5 → coverage = 1.0
3. Briefing answering 0/5 → coverage = 0.0
4. Cache hit: same briefing → returns from cache, no Haiku call
5. Malformed JSON from Haiku → returns `{error: ..., coverage: 0.0}` (conservative default)
6. Briefing < 100 chars (likely a fallback stub) → coverage = 0.0 without LLM call

**Cost.** ~$0.005 per evaluation with Haiku. Caching makes repeated canary use of the same baseline briefing free.

---

### U2 — `agent/canary_score.py`: composite verdict

**Goal.** Pure function that consumes `{baseline, candidate}` briefings + traces and emits a composite score. No I/O, fully unit-testable.

**Files.**
- `agent/canary_score.py` (new)
- `tests/test_canary_score.py` (new)

**Approach.** Single function:

```python
def composite_score(briefing, eval_record, weights=DEFAULT_WEIGHTS) -> dict:
    """Returns {composite, breakdown: {structure, operator, efficiency, errors}}."""
```

`eval_record` is the full eval shape (structure + trace_issues + judge + operator_sim). The function combines components into one float in [0, 1] for ordering and supplies the breakdown for the audit log.

Verdict thresholds:
- `delta > 0.05` → `promote`
- `delta < -0.10` → `discard`
- otherwise → `gate`

These are anchored to *noticeable* change, not any change — reduces noise-driven flapping.

**Test scenarios.**
1. Identical baseline/candidate → delta = 0.0 → `gate`
2. Candidate 50% fewer tokens, same structure, same operator-sim → delta > 0.05 → `promote` (THE case the saturated metric failed today)
3. Candidate same structure, operator-sim drops 0.4 → delta < -0.10 → `discard`
4. Candidate +1 summarize error, otherwise neutral → delta penalized by err weight → likely `discard`
5. Weights override: passing custom weights changes the verdict on borderline cases

---

### U3 — `agent/evaluate.py`: few-shot demonstrations replace lessons block

**Goal.** Replace `lessons_block()` output with concrete (prompt, behavior, score) demonstrations sourced from kept/reverted runs.

**Files.**
- `agent/evaluate.py` (modify): new `load_few_shot_examples()` + `demonstrations_block()`; `lessons_block()` deprecated but kept for backward compat
- `agent/loop.py` (modify): switch the SYSTEM_PROMPT append from `lessons_block()` to `demonstrations_block()`
- `tests/test_evaluate.py` (modify): new tests for few-shot extraction

**Approach.** `load_few_shot_examples(n_good=2, n_bad=1)`:
1. Read `edit_history.jsonl`; filter to `kept=True` (good) and `kept=False` (bad)
2. For each, look up the associated agent run in `runs.jsonl` by timestamp proximity
3. Extract: prompt, tool-call sequence, briefing snippet (first 200 chars), composite score (if available)
4. Format compactly:

```
## Good runs (compounding from kept edits)
[2026-05-27 score=0.78]
Prompt: "Competitive intel on Carbyne..."
Trace: 3 searches → 5 fetches → 1 summarize
Briefing snippet: "TL;DR: Both RapidDeploy and Carbyne saw..."

## Anti-patterns (avoid these)
[2026-05-26 score=0.20]
Prompt: "Top 3 stories..."
Trace: 8 searches → 9 fetches → 3 failed summarize → inline fallback
```

**Test scenarios.**
1. Empty edit_history → empty demonstrations block (degrades to current SYSTEM_PROMPT)
2. One kept + one reverted → demonstrations block contains both, correctly labeled
3. Missing matching run in runs.jsonl → skip silently, do not crash
4. Token-budget cap: long briefings truncated to 200 chars, full output stays under 500 tokens
5. Backward compat: `lessons_block()` still callable; importers continue to work

---

### U4 — `agent/code_edit.py`: code-edit sandbox

**Goal.** Apply a unified diff to a scratch copy of the repo, run pytest, return pass/fail + score. Never touches the live tree until human-approved.

**Files.**
- `agent/code_edit.py` (new)
- `tests/test_code_edit.py` (new)

**Approach.** Use a tempdir-based scratch repo:

```python
def evaluate_code_edit(diff_unified, test_diff_unified=None, run_pytest=True) -> dict:
    """
    1. Copy repo (minus .venv, data/) to a tempdir
    2. Apply diff_unified via `patch` or python's difflib.parse_unified
    3. If test_diff_unified: apply it too
    4. If run_pytest: run pytest in tempdir, capture pass/fail + output
    5. Optionally run a canary agent invocation in the scratch repo (deferred)
    6. Clean up tempdir
    Returns: {applied: bool, pytest_passed: bool, pytest_output: str, error: str|None}
    """
```

**Test scenarios.**
1. Trivial diff (rename a variable) → applied, pytest passes
2. Diff that breaks syntax → applied fails OR pytest fails — caught
3. Diff that doesn't apply (anchor mismatch) → applied=False, clear error
4. Diff with test_diff that exercises the change → both apply, pytest passes
5. Sandbox isolation: live repo files are byte-identical after run regardless of outcome

**Critical safety.** This module NEVER mutates the live tree. All operations happen in tempdir. The orchestrator decides what to do with the verdict.

---

### U5 — `agent/meta_eval.py`: wire reframings into propose + canary + autonomous

**Goal.** Update `META_EVAL_PROMPT` to allow code-type edits. Update `run_canary` to dispatch by edit type. Update `autonomous` to always-gate code edits.

**Files.**
- `agent/meta_eval.py` (modify): `META_EVAL_PROMPT` adds code-edit schema documentation + examples; `run_canary` branches on `edit.type`; `autonomous` recognizes code edits as always-gate
- `tests/test_meta_eval.py` (extend): new tests for code-edit canary path

**Approach.** Three changes:

1. **META_EVAL_PROMPT extension.** Add a section explaining when to propose code edits vs prompt edits:
   - Tool-contract bugs (e.g., schema issues) → propose code edit
   - Strategy / verbosity / formatting → propose prompt edit
   - Prefer minimal diffs; never modify tests in ways that mask the bug

2. **run_canary dispatcher.**
   ```python
   if edit["type"] == "code":
       return run_code_canary(edit)
   elif edit["type"] in ("add", "replace", "delete"):
       return run_prompt_canary(edit)
   ```
   `run_code_canary` calls `code_edit.evaluate_code_edit()` + (if pytest passes) runs `replay.replay_run` against a cached canary prompt with the patched code in the sandbox.

3. **autonomous hard-gate.** Any code edit returns `gate` from `classify_canary` regardless of canary outcome. Reason field: `"code edit: always gates for human review"`.

**Test scenarios.**
1. Mocked propose returns code edit → autonomous gates, no live mutation
2. Code edit with passing pytest + improved score → canary record shows promote-eligible, but classify returns `gate` (override fires)
3. Code edit with failing pytest → canary record shows `discard`
4. Prompt edit still flows through the prompt path (no regression)
5. Edit history records `type` (code|add|replace|delete) so future propose calls see it

---

### U6 — Integration & live demo verification

**Goal.** End-to-end run of the reframed system. Confirm the saturated-metric collapse is fixed AND code edits gate correctly.

**Files.** None — runtime verification only.

**Approach.**
1. **Confirm U0 fix lands.** Run agent on a fresh prompt. Verify no `summarize_missing_documents` trace_issue.
2. **Confirm composite verdict works.** Run autonomous mode (Haiku). The first proposed edit that gives token reduction should now `promote` (today it gated).
3. **Confirm operator-sim signal.** Inspect eval records — they now include `operator_sim.coverage` field.
4. **Confirm few-shot demos.** Run `python -m agent.meta_eval status` — output should show demonstrations, not abstract lessons.
5. **Confirm code edit path.** Trigger a propose where Haiku is biased toward a code suggestion (e.g., by adding a hint in META_EVAL_PROMPT). Verify the autonomous mode gates and writes a complete audit including diff + pytest result.

**Cost budget.** ~$0.40-0.60 for the full verification run (5-6 Haiku invocations + 1-2 full agent runs).

---

## Risks & mitigations

- **R1 — Operator-sim is itself an LLM and could be miscalibrated.** Two LLMs (sim + judge) → noise compounds. Mitigation: composite verdict weights structure_score and token-delta at 40% combined; even if operator-sim is noisy, the verdict isn't dominated by it.
- **R2 — Few-shot examples could leak prompts that ARE the bug.** If we promote a kept-edit that was actually lucky, the demonstration teaches the wrong behavior. Mitigation: only promote when composite delta > 0.05; raise this threshold over time if pattern leaks appear.
- **R3 — Code edits in a scratch tempdir might miss environment-specific failures.** Pytest passes in tempdir but fails in CI/prod. Mitigation: tempdir copies `.env` (without secrets), `pytest.ini`, and other config; CI-level validation is Tier 3.
- **R4 — Cumulative cost spike.** Each canary now does operator-sim too. Mitigation: cache operator-sim by briefing hash; one canary pair = baseline_op_sim + candidate_op_sim = ~$0.01 added overhead.
- **R5 — Backward compatibility break.** Existing eval records don't have operator_sim or composite fields. Mitigation: defensive reads; missing fields default to neutral (no penalty/bonus).

---

## Verification gates

1. All U0-U5 unit tests pass (`pytest tests/test_*.py`)
2. U0 verified live: `summarize_missing_documents` absent from new eval records
3. U2 verified: `composite_score` correctly orders the case that today's structure-only canary mis-gated (token reduction without quality loss → promote)
4. U3 verified: SYSTEM_PROMPT at runtime contains demonstrations, not abstract bullets
5. U4 verified: code edit applies in scratchdir, pytest runs, live tree unchanged
6. U5 verified: autonomous run with a code edit always returns `gate` even if canary signals strong
7. U6 verified: real autonomous cycle now produces a `promote` verdict on the historically-mis-gated case
8. Existing test suite still passes (110/110 + new tests)

---

## Build order (recommended)

Riskiest first when build cost is similar; sequential when each unit unlocks the next.

| # | Unit | Why now | Est. build |
|---|------|---------|------------|
| 1 | **U0** (summarize bug fix) | Removes noise from all downstream eval data | 30 min |
| 2 | **U1** (operator-sim) | Single highest-impact addition — solves saturated-metric collapse | 2-3 hr |
| 3 | **U2** (composite verdict) | Tiny wrapper, makes U1 actually drive verdicts | 1 hr |
| 4 | **U3** (few-shot demos) | Independent of U4; ships compounding | 1.5 hr |
| 5 | **U4** (code edit sandbox) | Highest risk, most novel; tempdir/patch infrastructure needs care | 3-4 hr |
| 6 | **U5** (wiring) | Glue; small | 1 hr |
| 7 | **U6** (live demo) | Verification only | 30 min runtime + $0.40-0.60 cost |

**Total estimated build:** ~10-12 hours of code + tests. **Verification cost:** under $1 of remaining Anthropic budget.

For interview defense: shipping U0+U1+U2 alone (≈4 hours) lands the strongest single answer to "your metric was wrong." U3 is cheap follow-up. U4+U5 is the genuinely novel piece worth a longer session.

---

## Deferred (Tier 3 — explicitly skipped per user direction)

- LangFuse / OpenTelemetry observability
- Statistical significance testing (K=10+, hypothesis tests)
- Production A/B testing on 5% of real runs
- Per-prompt-type lesson/example scoping
- Multi-tenant deployment
- Streaming live dashboard for autonomous mode

---

## Next step

Plan at `docs/plans/2026-05-27-002-feat-meta-eval-tier2-reframings-plan.md`. Ready to execute.

Recommended start: **U0** (30-minute prerequisite that eliminates the dominant noise pattern from all subsequent measurements). Then U1+U2 as the strongest single demo addition.
