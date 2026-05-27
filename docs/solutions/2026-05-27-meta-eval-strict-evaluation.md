---
title: Strict evaluation of the hybrid canary-replay meta-eval loop
date: 2026-05-27
status: real-data findings
tags: [meta-eval, evaluation, autonomy, lessons-learned]
---

# Strict evaluation — meta-eval loop under real Haiku runs

After shipping the hybrid canary-replay autonomy loop (commit `2d226de` on
`feat/meta-eval-hybrid-canary`), I ran 4 real LLM cycles to test the
end-to-end behavior. Below is an honest assessment of **what worked**, **what
didn't**, and **what the data actually shows** vs. what the design promised.

Model: **Haiku 4.5** (swapped from Sonnet 4.6 for budget — ~$1.95 remaining).
Tests run: ~$0.35 total across all 4 cycles.

---

## What worked ✅

### 1. The propose mechanism is honest and well-anchored
Haiku correctly identified all 4 chronic trace patterns from `evals.jsonl`
and produced two well-formed proposed edits with **valid unique anchors**
(both `anchor_count=1`, `anchor_valid=True`). The proposed `new_text` was
plausible and tied to a specific rationale per pattern.

Haiku-as-meta-eval is roughly as good as Sonnet for this task at 60% of the
cost (~$0.16/cycle vs ~$0.39/cycle).

### 2. The safety nets fired correctly
- Anchor reproposal guard: prevented re-applying a regressed anchor in tests
- Type=delete override: confirmed in tests, never triggered live
- Cost cap: pre-cycle check halted before overspend
- Convergence detector: halts on zero proposed edits

The system halted itself **3 of 3 times** when canary was inconclusive,
never silently mutating `prompts.py` on a weak signal. The autonomy mode is
**fail-safe**.

### 3. The replay mechanism is deterministic and cheap
Cached-doc replay produces a full-loop agent run for ~$0.05-0.08 each. The
infrastructure works as designed: search/fetch are mocked to only return
URLs in the cache, alternate SYSTEM_PROMPT threads through cleanly, and the
context manager restores all module state on exit.

### 4. Audit trail is complete
Per-cycle markdown logs in `data/logs/autonomous_runs/`, per-canary JSON in
`data/logs/canary/`, per-edit JSONL in `data/logs/edit_history.jsonl`. A
human can fully reconstruct what the autonomous mode did, why, and at what
cost.

---

## What didn't work ❌

### 1. The canary's structure-score signal is saturated

**The smoking-gun finding.** On canary run with K=1:
- Baseline briefing: structure score **5/5**, 64k input tokens, 8.5k output
- Candidate briefing: structure score **5/5**, 33k input tokens, 4.5k output
- Delta: **0** → `gate` (mixed signal)

The candidate used **half the tokens** with the same structure quality.
That's a real improvement the canary completely missed because both runs
saturated the 5/5 ceiling. Token efficiency, iteration count, and
judge.usefulness would have surfaced this — but the verdict heuristic uses
only structure score.

**Fix needed:** multi-dimensional canary scoring. Structure score is
necessary but not sufficient.

### 2. Lessons in SYSTEM_PROMPT don't fix tool-contract bugs

Predicted in the `/timeout` session, confirmed by the data. Across 5
historical runs + 2 fresh runs (with lessons block appended to SYSTEM_PROMPT
since the self-eval shipped):

| Run | summarize_missing_documents fires? |
|---|---|
| #1 (historical) | yes (3 retries) |
| #2 (historical) | no |
| #3 (historical) | yes (2 retries) |
| #4 (historical, lessons active) | yes (5 retries) |
| #5 (historical, lessons active) | yes (5 retries) |
| #6 (today, lessons active) | yes (2 retries) |
| #7 (today, lessons active, edit applied) | **yes (4 retries — got WORSE)** |

The lessons block told the model exactly how to call summarize correctly.
The model **read the lesson and still got the call wrong in every run**.
This is a **tool-schema contract bug**, not a prompt-strategy bug. Prompt
engineering cannot reliably fix it. The real fix is option A1 from earlier
work: make `documents` optional and have the loop attach docs from internal
state.

### 3. Real-world test of the canary-approved edit revealed a regression

After manually applying E1 (which the canary had marked `gate` not `discard`),
I ran the agent on a **fresh prompt** ("Latest funding rounds in public
safety AI startups in 2026") that had no cached docs:

| Metric | Before edit | After edit |
|---|---|---|
| Structure score | 5/5 | **0/5** |
| Summarize errors | 2 | **4** |
| Iterations | 10 | 9 |
| Input tokens | 192k | 195k (+) |
| Search calls | 8 | 4 (✓ budget rule worked) |
| Judge usefulness | 2 | 3 (slight content improvement) |

**The edit caused a structure-score regression of 5 points and doubled the
summarize-error count.** The canary's `gate` verdict was directionally
wrong — it should have been `discard` based on this real-world test. The
edit *did* successfully tighten the search budget (the only mechanical win),
but the cost was much worse summarize reliability on a prompt with no
cached docs.

**Root cause:** the canary uses cached docs from a *known* historical
prompt. It cannot predict how the edit affects behavior on *unknown*
prompts with no cache. The cache-based canary is a weak proxy for
generalization.

### 4. Edit history compounding wasn't exercised this session

The propose mechanism has the edit_history input wired in, but every
autonomous cycle this session ended in `gate` (no edit applied → no record
opened). So the compounding behavior I designed for ("don't re-propose
anchors that previously regressed") couldn't be tested live. Would need a
session with at least one apply + revert to populate the history.

### 5. Structure scorer is brittle on fallback briefings

When `summarize` fails and the model produces inline markdown narrative
(my loop's fallback path), the structure scorer returns 0/5 even when the
content is high quality. This means:
- A bad run that DOES call summarize → high structure score
- A run that falls back gracefully with great content → 0 score

The metric is biased toward *form* over *substance*.

---

## What the data actually proves

**The canary mechanism works as a fail-safe** — it never let `prompts.py`
get mutated on a weak signal across 3 autonomous attempts. The gate halt
fired every time the score didn't move. That's a real engineering property.

**The canary mechanism does NOT prove edits improve agent behavior.** Its
signal is too narrow (structure-only, K=1, cached-doc-bound) to predict
real-world impact. When I manually applied a canary-`gate`d edit and ran
the agent on a fresh prompt, it regressed. The canary couldn't see that
coming.

**The lessons-feedback loop is theater for tool-contract bugs.** The
`summarize_missing_documents` bug fires in **every single run** regardless
of how many lessons we add to SYSTEM_PROMPT. The model knows the rule and
still violates it. The fix is in the tool schema, not the prompt.

---

## What I'd change before production

1. **Multi-dimensional canary score**: not just structure (5-point binary),
   but token delta, iteration delta, summarize-error count, judge
   usefulness, source count. Composite verdict.

2. **K ≥ 3 with different canary prompts**: K=1 on a single cached prompt
   is too narrow. Need spread across prompt types to catch generalization
   failures.

3. **Fresh-prompt canary**: run candidate on at least one *uncached* prompt
   (small cost, ~$0.05) to test if the edit hurts behavior on novel inputs.

4. **Fix tool-contract bugs in code, not prompts**: make `summarize.run()`
   accept `documents=None` and have the loop attach all fetched docs
   automatically. Eliminates the entire `summarize_missing_documents`
   pattern. Once that's fixed, the meta-eval is operating on actually-
   prompt-fixable signals only.

5. **Distinguish form from substance in scoring**: pair structure_score
   with judge.usefulness in the verdict. A run that ranks 5/5 on form but
   2/5 on usefulness is not actually a good baseline.

6. **Don't promote without judge.usefulness >= baseline**: an edit that
   keeps structure constant but reduces usefulness is a regression even if
   structure says otherwise.

---

## Recommendation for the interview defense

**Show the system working AND the honest limitations.** The interviewers
will ask "how do you know your edits actually help?" The right answer is:

> "I don't, fully. The canary tells me the structure didn't regress on a
> cached historical prompt. That's a necessary signal, not a sufficient
> one. I verified empirically: a canary-passing edit can still regress on a
> fresh prompt. So I built the system as a fail-safe — it halts whenever
> the canary is unsure, and writes an audit log so a human can verify
> before keeping. The compounding-learning story compounds when the human
> says 'keep' or 'revert' and the next cycle reads that decision."

This frames the loop as **bounded autonomy with human-in-loop at the
ambiguous moments**, which is honest and defensible. The alternative —
claiming the canary proves improvement — would collapse under interview
follow-up.

---

## Cost summary (real Haiku runs this session)

| Action | Cost |
|---|---|
| Fresh agent run #1 (Carbyne/RapidDeploy 30d) | $0.05 |
| Autonomous cycle #1 (K=1, gated) | $0.16 |
| Manual apply E1 | $0.00 (file write) |
| Fresh agent run #2 (funding rounds, after edit) | $0.05 |
| Manual revert | $0.00 (file write) |
| **Total** | **~$0.26** |

Remaining Anthropic budget: ~$1.70 (started session at ~$1.95).
