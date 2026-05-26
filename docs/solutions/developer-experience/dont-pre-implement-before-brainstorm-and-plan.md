---
module: compound_engineering_workflow
date: 2026-05-26
problem_type: developer_experience
component: development_workflow
severity: high
applies_when:
  - "Engineer starts a CE-style workflow (brainstorm → plan → work) with full context already loaded"
  - "AI assistant is asked to 'help me prepare for X' with rich background material"
  - "Take-home assignment, interview prep, or any time-constrained build with strategic decisions"
tags:
  - compound-engineering
  - workflow-discipline
  - ai-assisted-development
  - ventriloquism
related_components:
  - tooling
---

# Don't pre-implement code or pre-author defense scripts before brainstorm/plan establishes shared decisions

## Context

When an AI assistant has rich background context (prior research, plans, interview prep), it's tempting to "just build" the solution end-to-end before walking through the CE phases (brainstorm → plan → work). The assistant infers what the user "probably wants" and produces implementation + defense scripts in one shot.

This **looks efficient** — code, README, planning docs, defense talking points all appear in 10 minutes. But it has a specific failure mode: **ventriloquism**. The artifact reads as if the user authored it (first-person voice, defended decisions, owned choices), but the user never validated any of those choices. The user is now in the position of defending decisions they didn't make.

In a high-stakes scenario (e.g., a live technical interview), this collapses on the first follow-up question that probes the candidate's actual reasoning rather than the surface answer.

## Guidance

When the user invokes a CE workflow, **stay in the workflow's interrogative phase** even if you can already write the output:

1. **Brainstorm** — produce a requirements doc, not implementation code. Ask clarifying questions to surface user preferences, constraints, and non-obvious tradeoffs.
2. **Plan** — produce an implementation plan with units, dependencies, and tests. Still no implementation code.
3. **Work** — execute the plan, one unit at a time, with explicit per-unit verification.

If you skip ahead, the user is forced into a corrective rollback that wastes the time you "saved" and often more. The signal is clear: when the user says "I want to do this CE-style," they want the dialog, not the deliverable-first.

## Why This Matters

The CE workflow's value is not just structured output — it's the **shared decision provenance**. Each fork the brainstorm/plan surfaces is a decision the user explicitly weighed in on. That ownership compounds in two ways:

1. **Defense readiness.** During code review, live interview, post-incident analysis — the user can explain *why* every load-bearing decision was made because they made it. Pre-authored defenses fail under "but why did you choose X over Y?" follow-ups.
2. **Refinement bandwidth.** Brainstorm/plan output is cheap to revise. Implementation + tests + docs are expensive. Front-loading interrogation makes scope cuts and pivot decisions trivially cheap before they become expensive.

The cost of staying interrogative: roughly +30-60 min of dialog before any code lands. The benefit: every line of code is one the user can defend without rehearsal.

## When to Apply

- Any CE skill invocation (`/ce-brainstorm`, `/ce-plan`, `/ce-work`, `/ce-doc-review`)
- Any time the user explicitly mentions "let's walk through this" or "I want to drive"
- Take-home or interview prep where the user will defend the output live
- When the assistant has rich context (recent research, prior session, loaded plan) — the temptation to skip ahead is highest exactly when the cost of skipping is highest

**Exception:** when the user explicitly says "just build it" or "no questions, just ship," honor that — but lower confidence in the resulting artifact's defensibility. Suggest a fast review pass afterwards.

## Examples

**Wrong (real session this learning came from):**

User: "I have an interview in 2 days, here's my prep folder + assessment, help me prepare"

Assistant immediately produced:
- 11 strategy/prep docs (interviewer profiles, defense talking points, scope decisions)
- Full implementation (~10 code files, working agent)
- README with inlined design decisions
- All in first-person voice as if user authored it

User reaction: "откати всё, мы пройдем через классический флот" (roll back everything, let's go through the classical flow). The hour spent on pre-authored implementation was wasted; the rollback + brainstorm restart took another 30+ min before useful work resumed.

**Right (subsequent recovery in same session):**

User: "now let's do the full ce cycle"

Assistant ran `/ce-brainstorm` → 6 single-question dialog turns (goal, stack, scope, anti-patterns, etc.) → requirements doc. Then `/ce-plan` → plan doc. Then `/ce-work` → implementation, one unit at a time with user visibility. Every load-bearing decision (Python over TS, single Sonnet, behavior tests, scope cuts) had user buy-in before code was written. Defense for live interview is genuinely owned.

## Prevention

- **When the user invokes a CE skill, follow its dialog phase fully — even if you could pre-empt the answers.** The dialog is the deliverable's foundation, not a formality.
- **Watch for the "I have context, let me build" temptation.** It's strongest at session start when prep material is fresh. That's exactly when staying interrogative pays the most.
- **If you find yourself drafting implementation code before a brainstorm question has been asked, stop.** That's the ventriloquism warning sign.
