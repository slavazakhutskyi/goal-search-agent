---
module: agent_runtime
date: 2026-05-26
problem_type: tooling_decision
component: development_workflow
severity: medium
applies_when:
  - "Building a multi-prompt agent demo on a fixed API budget (take-home assignment, prototype, demo deck)"
  - "Need to commit real example outputs but full-budget runs would exhaust the spending limit"
  - "Single-model architectural decision (D3-style 'single Sonnet') is correct for production, but artifact generation has different economics"
tags:
  - llm-cost-discipline
  - anthropic-pricing
  - take-home
  - artifact-provenance
  - haiku-vs-sonnet
related_components:
  - tooling
  - documentation
---

# Mixed-model briefing artifacts: Sonnet for the headline, Haiku for the rest, transparent provenance

## Context

Building a CLI agent demo with N example prompts on a fixed API budget (e.g., $5 take-home, $10 prototype). Running all prompts on the production model (Sonnet) exhausts the budget on artifact generation alone, leaving nothing for iteration, debugging, or live demo runs.

Naive options:
- **All Sonnet** — best output quality, blows the budget on one round
- **All Haiku** — fits budget easily but the artifact doesn't represent production model output
- **No real examples** — skip committed artifacts entirely, lose the "taste" signal in README

None of these match the realistic constraint of a budget-bounded demo.

## Guidance

**Use a mixed-model strategy: production model (Sonnet) for the headline artifact, cheaper model (Haiku) for the rest, with transparent provenance.**

Implementation:

1. **Default code uses production model.** `SONNET_MODEL = "claude-sonnet-4-6"` hardcoded in the LLM wrapper. The deployed agent is single-model, matching the production decision (D3 in the brainstorm sense).
2. **Generate the headline artifact (most important example) on production model.** This is the briefing inlined in the README, the example shown first.
3. **Generate the remaining artifacts on Haiku via temporary code swap.** Document the swap in a TEMP comment, revert before commit.
4. **Embed model name in artifact filenames.** Pattern: `{ts}-{model}-{slug}.md` (e.g., `20260526-143055-sonnet-give-me-a-briefing-on-rapidsos.md`, `20260526-141619-haiku-find-any-press-releases.md`). Provenance is visible at file listing level.
5. **Acknowledge the mixed set honestly in README.** Don't hide that two models were used. Frame as cost-discipline + incidental cross-model loop validation.

Sample README framing:

> *The default deployed model is Sonnet (`agent/llm.py:SONNET_MODEL`). The mixed-model set exists for two reasons: (1) honest cost-discipline — a single take-home budget shouldn't burn the whole API balance to demonstrate that the loop works on every prompt; (2) it incidentally demonstrates the agent produces structurally-valid briefings on either model.*

## Why This Matters

The naive "all Sonnet" approach trades budget headroom for a marginal quality lift on artifacts the reviewer may not even compare side-by-side. The naive "all Haiku" approach misrepresents what the production agent produces. The naive "no examples" approach loses the most visible quality signal in the README.

The mixed strategy:
- **Optimizes spend per signal.** Production-quality on the inlined example (where it matters most for taste signal), cheaper on the rest (where it matters for completeness).
- **Demonstrates cost discipline as a positive signal.** Reviewers familiar with Anthropic pricing (most engineering interviewers) will recognize the deliberate choice rather than reading it as "couldn't afford full quality."
- **Validates the agent's robustness across models.** Incidentally shows the tool-call loop and prompt templates work cleanly on both Sonnet and Haiku without code changes — a real signal about agent portability.

## When to Apply

- API budget is meaningfully smaller than full-Sonnet-on-all-prompts cost
- Artifact set has a natural "headline" (one prompt is most important for the README example)
- You're willing to be transparent about the mixed origin — opacity here backfires when a reviewer asks
- The code itself stays single-model for the production-deployment story

**Don't apply when:**
- Budget supports full-Sonnet generation with comfortable headroom
- All artifacts have equal weight (no natural headline)
- The deployment context demands single-model purity (e.g., highly regulated environment where mixed-model audit trails are a problem)

## Examples

Concrete numbers from one session (4 spec prompts, Claude Sonnet 4.6 + Haiku 4.5):

| Prompt | Model | Iterations | Tokens (in/out) | Approx cost |
|---|---|---|---|---|
| #1 RapidSOS 7-day (headline) | Sonnet | 12 (cap) | ~75K / ~7K | ~$0.33 |
| #1 RapidSOS 7-day (earlier Haiku smoke) | Haiku | 8 (cap) | ~50K / ~3K | ~$0.07 |
| #2 Top-3 stories | Haiku | 5 | ~66K / ~2.6K | ~$0.08 |
| #3 Competitors | Haiku | 6 | ~78K / ~7K | ~$0.12 |
| #4 Sentiment | Haiku | 6 | ~78K / ~7K | ~$0.12 |
| **Total** | mix | — | — | **~$0.72** |

Full-Sonnet equivalent of all 5 runs would be ~$3-4. The mixed approach stays comfortably under $1 while delivering one Sonnet artifact (the inlined README example) + four Haiku artifacts validating the loop on every prompt.

## Prevention

- **Plan model strategy explicitly in the brainstorm/plan phase**, not mid-build under cost panic
- **Track per-run tokens in `runs.jsonl`** so cost decisions are evidence-based
- **Add filename convention from day one** (`{ts}-{model}-{slug}.md`) — retrofitting provenance after the fact is tedious
- **Document the mixed-model decision in README** before pushing — silence here looks like an accidental compromise rather than a deliberate choice
