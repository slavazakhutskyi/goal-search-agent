# Plan — Meta-eval Tier 2 reframings: operator simulation + few-shot compounding

> Created: 2026-05-27 · Revised: 2026-05-27 (post doc-review) · Depth: Deep
> Parent plan: `docs/plans/2026-05-27-001-feat-meta-eval-loop-plan.md`
> Eval origin: `docs/solutions/2026-05-27-meta-eval-strict-evaluation.md`
> Mode: hybrid autonomy continues; this plan **replaces the scoring substrate** that hybrid mode uses.

> **Revision note (rev2).** Six reviewers (coherence, feasibility, product-lens, security-lens, scope-guardian, adversarial) converged on cutting U4 (code-edit sandbox) and U5's code-edit branch. Their independent reasoning: U0 already delivers the only concrete code-bug fix, U4's "always-gate" makes it functionally a `git stash && pytest` wrapper, and the 5 security findings (env-var leak, .env exposure, shell-injection in patch, audit-log integrity, meta_eval.py self-editability) are all mooted by dropping U4. Plan-level decisions applied here per `Apply convergent cuts + safe-auto fixes` routing. Active units: **U0, U1 (merged with U2), U3, U5 (prompt-path only).** Estimated build: **~5-7 hours** (was 10-12).

## Goal

The strict evaluation revealed three structural limits in the current system:
1. **The canary metric is saturated** — structure-score caps at 5/5, blind to token efficiency and content quality. A canary that should have promoted (50% token reduction, same structure) got gated.
2. **Lessons-as-rules don't transfer** — abstract bullets in SYSTEM_PROMPT do not reliably change behavior. The dominant bug (`summarize_missing_documents`) recurred in 6 of 7 runs *with the lesson active*.
3. **Prompts can't fix what code broke** — the most chronic failure is a tool-schema issue. The honest answer to this is **fix it in code directly** (U0), not build autonomous code-editing infrastructure.

This plan addresses (1) and (2) via reframings, and (3) via a direct human-authored code fix (U0). The result: **the canary measures something meaningful, lessons compound through demonstrations, and the dominant noise pattern is gone before any reframing runs.**

**Non-goals.**
- Autonomous code-editing meta-eval — *dropped from this plan after doc review*. The dominant bug is one human-authored patch (U0) away. Building a sandbox to propose code patches autonomously, when the only concrete motivating case is U0 itself, was solution-in-search-of-problem. The interview-defense story is stronger as "I made the surgical fix and explicitly chose not to build autonomous code-editing" than as "I built a sandbox the human still has to approve."
- Statistical significance testing (K=10+, hypothesis tests) — Tier 3.
- Production observability (LangFuse, dashboards) — Tier 3.
- Multi-tenant deployment — Tier 3.

---

## Two reframings (Reframing C dropped — see Non-goals)

### Reframing A — Operator simulation + composite verdict replaces structure-score

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

**Why this matters.** Operator-sim has higher resolution than structure-score (6 ordinal levels at coverage = N/5 vs structure's 6 levels at score/5), AND its dimension (utility-to-user) is orthogonal to structural form. Combined with token-efficiency and error-count in the composite, the canary now sees axes structure-only-scoring missed — like the 50%-token-reduction case from today's eval.

**Known limitation — adversarial flagged.** Operator-sim is *also* an LLM-based proxy. LLM judges asked "does this briefing answer this question?" default to generous YES on plausible-sounding text. A 5/5 coverage score on plausible nonsense is the failure mode to watch.

**Mitigation: groundedness sub-check.** For each answered question, the judge must also cite which `[^N]` footnote in the briefing supports it. Coverage counts ONLY questions whose YES answer cites at least one footnote. This collapses the "plausible nonsense" failure path — fabricated answers have no citation to point at.

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

**Known limitation — adversarial flagged.** The `kept=True / kept=False` labels in existing `edit_history.jsonl` were produced by the structure-only canary U1 is replacing. Surfacing them as demonstrations would teach the agent the old (broken) metric's biases.

**Mitigation: gate demonstrations behind N≥3 composite-scored runs.** `load_few_shot_examples()` returns an empty list until at least 3 records exist with `verdict` produced by the composite verdict. Until then, `demonstrations_block()` returns an empty string and SYSTEM_PROMPT is unchanged from the static base. Records produced under the new metric carry `metric_version: "composite-v1"` in `edit_history.jsonl`; the loader filters on this field.

**Matching rule (was underspecified).** When cross-referencing `edit_history.jsonl` to `runs.jsonl`, match by: **the most recent run in `runs.jsonl` whose timestamp is BEFORE the edit_history record's timestamp**. This is the run whose eval triggered the edit proposal. If no such run exists within a 24-hour window, skip silently.

---

## Architecture choice

One new module + extensions to existing ones:

- **`agent/operator_sim.py`** (new, ~120 lines) — operator simulation (question generation + answer judgment + groundedness check) **and** composite-verdict computation. Merged from the original U1+U2 split per scope-guardian review: composite verdict is a 4-line weighted formula with one consumer; separate module added no value.
- **`agent/evaluate.py`** (extend) — `load_few_shot_examples()` reads edit_history + runs.jsonl with the timestamp matching rule above, formats as demonstrations; replaces `lessons_block()` with `demonstrations_block()`; honors the `metric_version` filter.
- **`agent/meta_eval.py`** (extend) — `META_EVAL_PROMPT` unchanged in shape (still prompt-only edits); `run_canary` calls `operator_sim` + composite verdict instead of structure-only; `classify_canary` consumes composite score; new records carry `metric_version: "composite-v1"`.
- **`agent/tools/summarize.py`** (U0 edits) — `documents` arg becomes optional; loop auto-attaches.
- **`agent/loop.py`** (U0 edits) — pre-dispatch hook for `summarize` calls without `documents`.
- **`agent/prompts.py`** (U0 edits) — SYSTEM_PROMPT updated so the documented contract matches the new schema (avoids contradictory signals to the meta-eval).

Why one new module instead of two (was three): operator_sim and composite_score have one consumer each, share the canary's call path, and the composite formula is ~10 lines. Scope-guardian's argument was sound — splitting added boundary cost with no architectural gain.

---

## Pre-work: clear the noise floor

Before any of the reframings, fix the dominant code bug that's polluting eval data. Every reframing operates on signal-quality dimensions this bug currently dominates.

### U0 — Make `summarize.run(documents=None)` accept missing arg, with loop auto-attach + SYSTEM_PROMPT sync

**Goal.** Eliminate the `summarize_missing_documents` pattern entirely by making the tool contract permissive AND keeping the documented contract (SYSTEM_PROMPT) consistent with the new schema.

**Files.**
- `agent/tools/summarize.py` (modify):
  - Signature: `def run(prompt: str, documents: list[dict] | None = None)` → coerce `documents or []` internally
  - `TOOL_SCHEMA`: remove `documents` from `required`; update description to "documents optional; loop will attach all successfully-fetched docs if omitted"
- `agent/loop.py` (modify): **inside `run()`, NOT `_dispatch_tool()`** — the dispatcher is generic and has no `tool_calls` context. Reuse the existing fallback pattern at `loop.py:167-174` (which already filters cached docs by `text` and `error`). Extract it into a helper `_collect_fetched_docs(tool_calls)`. Before calling `_dispatch_tool` on a `summarize` tool_use that has no `documents` in its input dict, inject `documents=_collect_fetched_docs(tool_calls)` into the input.
- `agent/prompts.py` (modify): update SYSTEM_PROMPT Strategy step 4 from *"Call summarize ONCE with the original prompt and the fetched documents"* to *"Call summarize ONCE with the original prompt; the loop attaches fetched documents automatically if you omit the `documents` arg, but passing them explicitly is fine"*. Without this edit, the prompt contradicts the new schema and the meta-eval sees mixed signals.
- `tests/test_loop.py` (modify): add test for auto-attach behavior

**Approach.** Reuse the existing `loop.py:157-162` fetched-doc collection pattern. Extract to a helper, call it from the new pre-dispatch hook. The model can omit `documents` and the loop attaches what's actually been fetched this run. No behavior change for runs where the model passes `documents` correctly.

**Note on existing test (`tests/test_evaluate.py:68`).** This test asserts the trace_issue fires on the exact error string `"run() missing 1 required positional argument: 'documents'"`. After U0, that error is impossible to produce naturally. Update the test to assert the issue does NOT fire on the auto-attach path, while keeping a separate unit test that the detector logic still works when given a synthetic matching error string (for backward compat).

**Test scenarios.**
1. Model calls `summarize(prompt="X")` → loop auto-attaches fetched docs → summarize succeeds
2. Model calls `summarize(prompt="X", documents=[...])` → existing behavior unchanged (model-supplied docs win)
3. Model calls `summarize(prompt="X")` with zero successful fetches → empty list → empty-docs stub briefing
4. Trace detector: synthetic error string still fires the trace_issue (backward compat for old logs)
5. End-to-end: live agent run → next eval record has NO `summarize_missing_documents` in `trace_issues`

**Verification.** Run a full live agent invocation after this lands. The next eval record should show `summarize_missing_documents` is absent from `trace_issues`.

---

## Implementation units

### U1 — `agent/operator_sim.py`: operator follow-up simulation + composite verdict

**Goal.** One module containing both the operator-utility signal (Haiku call) and the composite verdict that consumes it. Merged from the original U1+U2 per scope-guardian review — composite verdict had one consumer and no separate test surface.

**Files.**
- `agent/operator_sim.py` (new, ~120 lines)
- `tests/test_operator_sim.py` (new)

**Approach.** Two public functions in one module:

```python
def coverage_score(prompt: str, briefing: str) -> dict:
    """One Haiku call. Returns {questions, answered, grounded_in_citation, coverage}."""

def composite_score(eval_record: dict, weights=DEFAULT_WEIGHTS) -> dict:
    """Pure function. Combines coverage + structure + token-delta + error-count."""
```

`OPERATOR_SIM_PROMPT` extended with the groundedness check:

```
You are a RapidSOS operator who has 3 minutes to triage <briefing>.

User prompt: {prompt}
Briefing: {briefing}

Step 1: List 5 follow-up questions you'd most want answered in your next 30 minutes.
Step 2: For each, answer YES/NO: does the briefing as-written already answer it?
Step 3: For each YES, cite the [^N] footnote that supports it. If you cannot
        cite a footnote, the answer must be NO (the briefing is implying
        without sourcing).

Return STRICT JSON:
{
  "questions": ["...", "...", ...],
  "answered": [true, false, true, true, false],
  "citations": ["[^2]", "", "[^4]", "[^1,3]", ""],
  "coverage": 0.4   # only YES with non-empty citation count
}
```

Coverage = `sum(answered[i] AND citations[i] != "") / len(questions)`. Defeats plausible-nonsense judge bias.

Caching: same briefing → same coverage score. Cache by `hash(briefing)` to avoid re-paying within one autonomous session.

Composite formula:
```
score = w_op * coverage             # 0..1, primary
      + w_struct * structure_score / 5
      + w_eff * clip(-token_delta_normalized, -1, 1)
      + w_err * clip(-error_count, -1, 1)
```
Default `w_op=0.5, w_struct=0.2, w_eff=0.2, w_err=0.1`. Verdict thresholds (anchored to noticeable change):
- `delta > 0.05` → `promote`
- `delta < -0.10` → `discard`
- otherwise → `gate`

**Contract migration (was F3 — undefined before).** The existing `classify_canary` in `meta_eval.py:731-749` uses integer structure-score deltas. **U1 REPLACES `classify_canary` entirely** — the composite function becomes the only verdict path. K-result records' shape changes from `{baseline_score: int, candidate_score: int, delta: int}` to `{baseline_eval: dict, candidate_eval: dict, composite_delta: float, breakdown: dict}`. Existing canary records in `data/logs/canary/` are tagged `metric_version: "structure-v0"` and remain on disk for historical reference but are NOT used by demonstrations or future verdicts. New records carry `metric_version: "composite-v1"`.

**Test scenarios.**
1. Mocked Haiku → returns coverage 0..1 with matching question/answer/citation lengths
2. Briefing answering 5/5 with all citations → coverage = 1.0
3. Briefing answering 5/5 with NO citations → coverage = 0.0 (groundedness gate caught the nonsense case)
4. Cache hit: same briefing → no second Haiku call
5. Malformed JSON from Haiku → `{error: ..., coverage: 0.0}` (conservative default)
6. Briefing < 100 chars → coverage = 0.0 without LLM call
7. Composite — identical baseline/candidate → delta = 0.0 → `gate`
8. Composite — candidate 50% fewer tokens, same coverage, same structure → delta > 0.05 → `promote` (THE case today's structure-only canary failed)
9. Composite — candidate coverage drops 0.4 → delta < -0.10 → `discard`
10. Composite — candidate +1 summarize error, otherwise neutral → delta penalized

**Cost.** ~$0.005 per coverage call. Caching makes repeated canary use of same baseline free. Per-canary overhead vs structure-only: ~$0.01.

---

### U2 — MERGED INTO U1 (composite verdict folded in)

Original U2 (`agent/canary_score.py` as a separate module) deleted per scope-guardian Finding 2. The composite formula is ~10 lines with one consumer; separate module added boundary cost without architectural gain. Test scenarios absorbed into U1's test file.

---

### U3 — `agent/evaluate.py`: few-shot demonstrations replace lessons block (with poisoned-label gate)

**Goal.** Replace `lessons_block()` output with concrete (prompt, behavior, score) demonstrations sourced from kept/reverted runs **scored under the new composite verdict**. Pre-composite records are excluded to prevent teaching old-metric biases.

**Files.**
- `agent/evaluate.py` (modify): new `load_few_shot_examples()` + `demonstrations_block()`; `lessons_block()` deprecated but kept for backward compat
- `agent/loop.py` (modify): switch the SYSTEM_PROMPT append from `lessons_block()` to `demonstrations_block()`
- `agent/meta_eval.py` (modify): when writing edit_history records, tag with `metric_version` field
- `tests/test_evaluate.py` (modify): new tests for few-shot extraction + version-gate

**Approach.** `load_few_shot_examples(n_good=2, n_bad=1, min_records=3)`:
1. Read `edit_history.jsonl`; filter to records with `metric_version == "composite-v1"`
2. If fewer than `min_records` (default 3) composite-scored records exist → return empty list (the poisoned-label gate)
3. Among remaining: filter to `kept=True` (good) and `kept=False` (bad); sort by composite score
4. For each, find the associated agent run via the timestamp-matching rule (most recent run BEFORE the edit_history timestamp, within 24h)
5. Extract: prompt, tool-call sequence, briefing snippet (first 200 chars), composite breakdown
6. Format compactly:

```
## Good runs (compounding from kept edits, scored under composite verdict)
[2026-05-27 score=0.78 op=0.8 struct=1.0 tokens=-30%]
Prompt: "Competitive intel on Carbyne..."
Trace: 3 searches → 5 fetches → 1 summarize
Briefing snippet: "TL;DR: Both RapidDeploy and Carbyne saw..."

## Anti-patterns (avoid these)
[2026-05-26 score=0.20 op=0.2 struct=0.0 tokens=+15%]
Prompt: "Top 3 stories..."
Trace: 8 searches → 9 fetches → 3 failed summarize → inline fallback
```

**Test scenarios.**
1. Empty edit_history → empty demonstrations block (degrades to base SYSTEM_PROMPT)
2. 2 records but both pre-composite (`metric_version="structure-v0"`) → empty demonstrations block
3. 3+ composite-scored records, mix of kept/reverted → demonstrations block populated
4. Missing matching run in runs.jsonl (no run within 24h before edit) → skip that record silently
5. Token-budget cap: long briefings truncated to 200 chars, full output stays under 500 tokens
6. Backward compat: `lessons_block()` still callable; importers continue to work

---

### U4 — DROPPED (code-edit sandbox)

Per doc review convergence (product-lens F3, scope-guardian F4, adversarial F2, security-lens 5 findings). The single concrete code-bug motivating this unit is fixed by U0 in 30 minutes. After U0, no second motivating case exists. The "always-gate" policy reduces autonomous value-add to "git stash && pytest" with a Haiku API call. Security findings (env-var leak, .env exposure, patch-shell-injection, audit-log integrity, meta_eval.py self-editability) all moot when U4 doesn't ship.

Defer to a post-interview session if and when a second code-level bug class emerges in eval data.

---

### U5 — `agent/meta_eval.py`: wire reframings (prompt-path only)

**Goal.** Wire operator-sim + composite verdict into the existing prompt-edit canary path. No code-edit branch.

**Files.**
- `agent/meta_eval.py` (modify): `run_canary` calls `operator_sim.coverage_score` + `operator_sim.composite_score` instead of structure-only scoring; `classify_canary` REPLACED with composite logic per U1 contract migration; new records tagged `metric_version: "composite-v1"`
- `tests/test_meta_eval.py` (modify): update existing tests to expect composite verdict shape

**Approach.** Three changes:

1. **No META_EVAL_PROMPT change.** The prompt-edit schema (add/replace/delete with anchor) is unchanged. Removing the code-edit branch keeps the proposer's surface tight.

2. **run_canary swaps scoring.** Replace structure-score evaluation with composite verdict from U1. Per-K-iteration record now carries `{baseline_eval, candidate_eval, composite_delta, breakdown, metric_version}` instead of int deltas.

3. **classify_canary replaced.** Old integer-delta version retired. New version reads composite_delta directly. Single decision path.

**Test scenarios.**
1. Mocked operator-sim returns coverage 0.8 for both → composite delta near 0 → gate
2. Candidate token-delta -30%, same coverage → composite delta > 0.05 → promote (the case today's canary missed)
3. Candidate coverage drops 0.4, structure unchanged → composite delta < -0.10 → discard
4. New canary records carry `metric_version: "composite-v1"`
5. Existing structure-only tests pass (after updating to new contract)

---

### U6 — DROPPED as a unit (folded into Verification gates)

Per scope-guardian Finding 5: U6 was runtime verification, not implementation. Its content is the verification gates section below.

---

## Risks & mitigations

- **R1 — Operator-sim is itself an LLM and could be miscalibrated.** Two LLMs (sim + judge) → noise compounds. Mitigation: composite verdict weights structure_score and token-delta at 40% combined; even if operator-sim is noisy, the verdict isn't dominated by it. **Additional mitigation (adversarial-flagged):** groundedness check requires each YES to cite a `[^N]` footnote, defeating plausible-nonsense judge bias.
- **R2 — Few-shot examples could leak prompts that ARE the bug.** If we promote a kept-edit that was actually lucky, the demonstration teaches the wrong behavior. Mitigation: only promote when composite delta > 0.05. **Additional mitigation (adversarial-flagged):** demonstrations gated behind ≥3 composite-scored records (`metric_version="composite-v1"`); pre-composite records excluded.
- **R3 — Cumulative cost spike.** Each canary now does operator-sim too. Mitigation: cache operator-sim by briefing hash; one canary pair = baseline_op_sim + candidate_op_sim = ~$0.01 added overhead. **Verification cost (feasibility F6):** total verification budget revised to ~$0.50-0.80 (was $0.40-0.60) for one autonomous cycle + 1 fresh agent run.
- **R4 — Backward compatibility break.** Existing eval records don't have operator_sim or composite fields. Pre-composite edit_history records have `kept` decided under the old metric. Mitigations: defensive reads (missing fields default to neutral); `metric_version` field distinguishes old vs new records; demonstrations gate on `metric_version="composite-v1"` (R2).
- **R5 — U0 contract change ripples.** Making `documents` optional affects `summarize.TOOL_SCHEMA`, `SYSTEM_PROMPT`, existing test `test_evaluate.py:68` (which asserts the exact error string), and the loop's pre-dispatch hook. Mitigation: U0's file list now explicitly enumerates all four touch points; test update preserves backward-compat for the detector logic.

---

## Verification gates (was U6; folded in here per scope-guardian Finding 5)

**Unit tests:**
1. All U0, U1, U3, U5 unit tests pass (`pytest tests/test_*.py`)
2. Existing test suite still passes (110/110 + new tests; existing test_evaluate.py:68 updated per U0 notes)

**Live verification (single autonomous cycle + 1 fresh agent run, budget ~$0.50-0.80):**
3. **U0 verified live:** run agent on a fresh prompt → `summarize_missing_documents` absent from the new eval record's `trace_issues`
4. **U1 composite verdict verified:** the historically-mis-gated case (token reduction with same structure) now produces `composite_delta > 0.05` → `promote`. Verify on the cached "top 3 public safety AI stories" canary that today's structure-only canary gated at delta=0.
5. **U1 groundedness verified:** inspect the operator_sim record — at least one `YES` answer has an empty citation field, demonstrating the gate is functional (not always-true).
6. **U3 demonstrations gate verified:** until ≥3 composite-scored edit_history records exist, `demonstrations_block()` returns empty string. SYSTEM_PROMPT runtime should be unchanged from base for the first 2 autonomous cycles after this lands.
7. **U5 contract migration verified:** new canary records carry `metric_version="composite-v1"`. Old records still readable but excluded from demonstrations.

**Verification budget:** ~$0.50-0.80 (revised from $0.40-0.60 per feasibility F6). Within the ~$1.70 remaining budget.

---

## Build order (recommended)

Riskiest first when build cost is similar; sequential when each unit unlocks the next.

| # | Unit | Why now | Est. build |
|---|------|---------|------------|
| 1 | **U0** (summarize bug fix + SYSTEM_PROMPT sync) | Removes the dominant noise pattern. Without this, U1/U3 measurements are unreliable | 45 min (was 30; SYSTEM_PROMPT edit + test update) |
| 2 | **U1** (operator-sim + composite verdict, merged) | Solves the saturated-metric collapse. Includes groundedness check + composite formula | 3-4 hr (was 2-3 + 1 separate) |
| 3 | **U3** (few-shot demos with poisoned-label gate) | Independent of U1's internals; the `metric_version` gate keeps it safe to ship in parallel | 2 hr (was 1.5; added gate logic) |
| 4 | **U5** (prompt-path wiring + classify_canary replacement) | Glue: replaces classify_canary with composite, swaps run_canary scoring path | 1 hr |
| 5 | Verification run | Single autonomous cycle + 1 fresh agent run; verify gates 3-7 | 30 min runtime + $0.50-0.80 cost |

**Total estimated build:** ~7-8 hours of code + tests (was 10-12). **Verification cost:** $0.50-0.80 of remaining $1.70 budget.

**Cut scope vs original plan:**
- U2 merged into U1 — saved 1h boundary cost, no architectural loss
- U4 dropped entirely — saved 3-4h, eliminated 5 P1 security findings
- U5 code-edit branch dropped — saved ~30 min, simplified wiring
- U6 dropped as a numbered unit — folded into Verification gates

For interview defense: this revised plan IS the strongest single answer. Each unit ties to a measured failure mode from today's strict eval. The dropped U4 IS the defensible story: *"I deliberately chose not to build autonomous code-editing — the dominant bug was one human-authored patch (U0) away. The interview-defense story is stronger as bounded autonomy with surgical code fixes than as a sandbox a human still has to approve."*

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

Plan revised post doc-review. Ready to execute.

Recommended start: **U0** (45-minute prerequisite that eliminates the dominant noise pattern from all subsequent measurements + syncs SYSTEM_PROMPT to the new contract). Then U1 (merged operator-sim + composite verdict) as the strongest single demo addition. U3 (with the metric_version gate) ships compounding without poisoning from old labels. U5 is glue.

**For interview defense:** the revised plan reduces surface area while delivering the same insight. The decision to cut U4 is itself the senior-engineer judgment story — "I noticed the meta-eval was tempting me to build autonomous code-editing, recognized that the dominant bug was one human patch away, and explicitly chose the surgical fix over the elaborate machinery."
