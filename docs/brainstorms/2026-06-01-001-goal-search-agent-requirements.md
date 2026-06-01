---
title: Goal-search agent — universal goal-completion search engine
date: 2026-06-01
status: requirements
source: ce-brainstorm session 2026-06-01
parent_pr: https://github.com/slavazakhutskyi/rapidsos-intel-agent/pull/1
---

# Goal-search agent

## Problem frame

Today's user (Slava + similar) does goal-directed research by hand: Google + Claude in a browser. Pattern is always the same — formulate a goal ("companies that fit my skills as freelance leads", "amateur football pitches in Berlin", "competitive intelligence on Carbyne+RapidDeploy+Prepared"), iterate through search results, copy-paste promising candidates, write short notes per candidate, decide when "enough." Cost per goal: 30-90 minutes of human attention. Output quality: inconsistent, no record of why each candidate was kept or dropped, no compounding across goals.

The existing `rapidsos-intel-agent` (this repo, PR #1) automates the final step — given named targets, produce a structured briefing. Two real-world use cases sit upstream of that step and are NOT covered: (a) **goal-from-criteria** discovery (find N candidates matching Y), and (b) **goal-completion judgment** (when is the list "enough").

This brainstorm reframes the repo from a single-domain briefing tool to a **generic goal-completion search agent** — promptable across domains (companies / places / products / people / arbitrary), with the meta-eval loop adapted to score goal-completion rather than briefing-structure-only.

**Primary user.** Slava himself (portfolio piece + personal research tool). Adjacent users (other developers running similar goal-driven searches) inherit value without explicit targeting.

## Actors

- **A1 — Goal-author.** Writes a single natural-language prompt expressing a goal. Examples in AE1-AE3 below. Knows what they want; does NOT want to learn DSL or config flags.
- **A2 — Agent.** Single LLM-driven loop. New tools augment the existing search/fetch/summarize: `add_candidate`, `score_candidate`, `finalize`. Decides when goal is satisfied.
- **A3 — Meta-eval (existing).** Reads runs/evals, proposes prompt edits, gates via canary, kept across this refactor. Adapted metric to score goal-completion.

## Key flows

- **F1 — Single-prompt goal completion.** User runs `python -m agent "<goal prompt>"`. Agent searches, fetches, scores candidates, finalizes when goal-completion deterministic threshold is reached (default: N≥10 candidates with score≥0.7) OR when LLM-judge says "enough" for non-standard goals.
- **F2 — Backward-compat briefing path.** User runs original-style prompt ("competitive intelligence on X, Y, Z"). Agent detects named-targets pattern, skips discovery, runs existing briefing pipeline. Output shape adapts (markdown briefing vs ranked candidates list).
- **F3 — Meta-eval over goal runs.** Existing meta-eval cycle (propose → canary → keep/revert) runs against the new goal-coverage metric. `evals.jsonl` records both legacy briefing-structure score AND new goal-coverage score; canary chooses based on output shape detected per run.

## Acceptance examples

- **AE1 — Freelance leads.**
  Prompt: `find 10 freelance leads for senior fullstack dev (python+react), remote-friendly, EU timezones, last 30 days`
  Expected output: markdown list of ≥10 candidates, each with: name, URL, why-fits (1-2 sentences grounded in fetched content), score 0-1. Sorted by score desc. Run completes under $0.30. Goal-coverage score ≥0.7.

- **AE2 — Places (radical domain shift).**
  Prompt: `find 10 best amateur football pitches in Berlin with online booking, open Sunday mornings`
  Expected output: same shape as AE1 — list of ≥10 candidates with name, URL, why-fits, score. Works without code change vs AE1. Proves universality.

- **AE3 — Backward-compat briefing.**
  Prompt: `competitive intelligence on Carbyne, RapidDeploy, Prepared`
  Expected output: existing structured briefing (TL;DR / themes / sentiment / sources). NOT a candidates list. Agent detects named-targets pattern and routes to legacy briefing path. Output identical in shape to existing PR #1 behavior on this prompt.

## Scope

### In scope

- New tools: `add_candidate(url, name, why_fits, score, axes?)`, `score_candidate(candidate_id, score, axes?)`, `finalize()`.
- Deterministic goal-completion check: count of candidates with `score >= COMPLETION_SCORE_THRESHOLD` (default 0.7); when count ≥ goal-N (parsed from prompt or default 10), set `goal_met=True`. Loop terminates next iteration.
- LLM-judge backup: when prompt does NOT parse cleanly to a numeric goal-N (e.g., "find a good restaurant" — single result, no count), the LLM is asked once per iteration "is this goal sufficiently met? answer yes/no with reason." Triggered only when deterministic threshold cannot be evaluated. Conservative: defaults `no` on ambiguity.
- Universal score: every candidate has a single `score` field in [0, 1], rubric = "relevance to stated goal." Optional `axes: dict[str, float]` field carries per-dimension breakdown when the goal naturally decomposes (e.g., `{"timezone_match": 0.9, "skill_fit": 0.8, "freshness": 0.6}`). LLM populates `axes` if useful; never required.
- Output detection: agent emits one of two shapes — `candidates_list` (AE1, AE2) or `briefing_markdown` (AE3). Detection rule: if `finalize()` was called with ≥3 candidates → candidates_list; if `summarize()` was the terminal call (no `finalize`) → briefing_markdown. Single per-run dispatch.
- Goal-coverage metric (new, in `agent/operator_sim.py` or sibling module): scores a `candidates_list` output by sampling N=5 LLM-judge probes asking "does this list cover the goal? are any candidates obviously low-quality? is the goal-N reached?" Returns coverage∈[0,1]. Briefing runs (AE3) keep existing composite (operator-sim + structure + efficiency + errors).
- Composite verdict in canary: detects output shape per run, applies briefing-composite OR goal-coverage accordingly. Single `metric_version` bump: `composite-v1` → `goal-v1`. Old records remain valid (loader uses defensive defaults).
- Repo rename: `rapidsos-intel-agent` → `goal-search-agent` (GitHub repo + Python package path stays `agent/` for diff minimization).
- README rewrite: replace RapidSOS framing with generic goal-search framing. AE1+AE2+AE3 included as worked examples.
- Branding cleanup in `agent/prompts.py`: remove competitor-specific rules (parent-company mapping); keep tool-use discipline; add goal-completion language.

### Out of scope (deferred for later)

- Hosted service / API / web UI.
- Multi-provider LLM support (stays on Anthropic Haiku/Sonnet).
- Domain-specific scoring rubrics beyond the universal score + optional axes.
- Persistent multi-session candidate stores (each run starts fresh).
- Auto-discovery of follow-up goals from prior runs.

### Outside this product's identity

- General LLM-powered chat assistant. This is goal-completion, not conversation.
- Lead-generation SaaS. The product is a CLI portfolio piece, not a commercial tool.
- Briefing-as-a-service. AE3 backward-compat is a preservation, not a primary offering.

## Decisions & rationale

- **D1 — Deterministic goal-completion check + LLM-judge backup.** Reliability primary, flexibility secondary. Deterministic check (N candidates ≥ score threshold) is cheap, predictable, debuggable. LLM-judge handles the ~20% case where the prompt is not numerically parseable. Eliminates the runaway-LLM termination risk while preserving universality. *(User confirmed 2026-06-01.)*
- **D2 — Universal score in [0,1] + optional axes dict.** Simplicity over precision. A universal scalar makes ranking trivial and metric design clean; the optional axes dict gives downstream consumers (and the LLM judge) richer context when the goal decomposes. Per-goal-type rubrics rejected as carrying-cost overhead disproportionate to benefit. *(User confirmed 2026-06-01.)*
- **D3 — Output-shape detection per run; preserve briefing path for AE3.** Single agent runtime, dual output shape. Existing meta-eval composite stays for briefing runs; new goal-coverage metric for candidates-list runs. Per-run dispatch by terminal-tool detection. Eliminates a separate `briefing` vs `goal-search` mode flag at the CLI surface — the prompt itself decides. *(Agent recommendation confirmed 2026-06-01.)*
- **D4 — Repo rename + branding cleanup.** Portfolio framing requires removing the take-home-specific signal. Repo name + README rewrite + competitor-mapping removal from prompts. Package path (`agent/`) preserved to minimize diff and keep git history clean. *(Agent recommendation; not surfaced as a call-out — low-stakes mechanical task.)*

## Success criteria

- **S1.** All 3 acceptance examples (AE1, AE2, AE3) work end-to-end on a single `python -m agent "<prompt>"` command without per-prompt flags. (Behavioral.)
- **S2.** Total cost per acceptance example ≤ $0.30 under Haiku 4.5. Verified live. (Operational.)
- **S3.** Meta-eval cycle runs on a candidates-list output (AE1 or AE2) and produces a non-trivial canary verdict (not "n/a"). (Architectural — proves the metric switch wired correctly.)
- **S4.** README contains generic framing + AE1/AE2/AE3 as worked examples. Zero RapidSOS-specific strings outside `docs/solutions/` and `docs/plans/` (which preserve historical context). (Portfolio-defensibility.)
- **S5.** Existing test suite (140/140 on `main`) plus new goal-search tests all pass. No regression in AE3 path. (Quality.)

## Dependencies / assumptions

- **Dep1.** Haiku 4.5 model continues to be available and priced at current rates. Agent budget assumes ~$1.50 remaining for end-to-end demo runs of AE1+AE2+AE3.
- **Dep2.** SearXNG local instance running at `localhost:8888` (existing infra; no change).
- **Asm1.** Prompts of the form "find N X with criteria Y" parse cleanly to a numeric goal-N via simple regex or LLM extraction. Falls back to default N=10 if no number present.
- **Asm2.** The shape-detection rule (terminal-tool call) is sufficient to route to briefing vs candidates output. Edge cases (model calls neither `finalize` nor `summarize`) treated as the existing partial-fallback path.

## Outstanding questions (deferred to planning)

- **OQ1.** Where does the shape-detection live — `loop.py` post-loop, or as a new dispatch module? *(Implementation detail, ce-plan resolves.)*
- **OQ2.** Goal-coverage metric exact prompt + judge model + sample size. Universal-score baseline says K=5 judge probes; not validated under load. *(Empirical tuning during execution.)*
- **OQ3.** Whether `score_candidate` exists as a separate tool or is folded into `add_candidate` (single-shot per candidate). Agent suspects folding is cleaner. *(ce-plan call-out.)*

## Next step

`/ce-plan` to expand into implementation units. Recommended depth: Standard. Budget: ~$1.50 remaining Anthropic credit. Estimated build: 5-8 hours for AE1+AE2 (new path) + ~30 min to verify AE3 backward-compat.

After plan lands, prefer iterative execution via `/ce-work` with live verification after each milestone.
