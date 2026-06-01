# Goal-Search Agent

A CLI agent that takes one natural-language goal, decides which output shape fits, and runs an LLM tool-use loop until the goal is met. Two output shapes, one prompt:

- **Ranked candidates list** — "find N <things> matching <criteria>"
- **Narrative briefing** — "analyze / summarize / report on <named targets>"

The prompt alone selects the shape. No flags, no config.

A self-evaluation + meta-eval loop layers on top: every run is critiqued, lessons feed back into the system prompt on the next run, and proposed prompt edits are gated by a cached-doc canary replay before any mutation lands.

---

## What it does

```bash
# Shape A — candidates list
python -m agent "find 10 freelance leads for senior fullstack dev (python+react), remote-friendly, EU timezones, last 30 days"
python -m agent "find 10 best amateur football pitches in Berlin with online booking, open Sunday mornings"

# Shape B — narrative briefing (named targets)
python -m agent "competitive intelligence on Apple, Microsoft, Google"
python -m agent "summarize sentiment of recent coverage of AI in education"
```

Each invocation:
1. Plans 1-3 focused web searches via a local SearXNG instance.
2. Fetches the most relevant URLs.
3. Either accumulates ranked candidates with per-candidate score + rationale (shape A) or produces a structured TL;DR + themes + sentiment briefing (shape B).
4. Logs the full run to `data/logs/runs.jsonl` for post-hoc analysis.
5. Generates a self-eval (`data/logs/evals.jsonl`) — structure check, trace critique, Haiku judge.

---

## Quick start

```bash
# Prereqs: Docker, Python 3.11+, an Anthropic API key

docker compose up -d                          # boots SearXNG on :8888
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                          # then add ANTHROPIC_API_KEY

python -m agent "find 10 best amateur football pitches in Berlin"
```

The output prints to stdout. A markdown copy lands in `data/briefings/`, and one line per run is appended to `data/logs/runs.jsonl`.

### Run the tests

```bash
pytest -v
```

All 182+ tests pass on a fresh checkout without an API key or running Docker — the test suite mocks the Anthropic client and SearXNG.

---

## How it works

**Loop** (`agent/loop.py`). Single Anthropic tool-use loop with 5 tools:

| Tool | Purpose |
|---|---|
| `search(query)` | SearXNG-backed web search |
| `fetch(url)` | URL → cleaned text + metadata (cached) |
| `add_candidate(url, name, why_fits, score, axes?)` | Record one candidate for shape A; loop upserts by URL |
| `finalize()` | Signal "goal met"; loop emits ranked list |
| `summarize(prompt, documents?)` | Shape B terminal — `documents` is optional, auto-attached from cache |

A goal-completion terminator runs each iteration: deterministic check on candidates above a score threshold + LLM-judge backup for non-numeric goals. When the goal is met a `GOAL_HINT` text block is injected into the model's next turn.

**Output-shape dispatch** (`agent/loop.py` post-loop). `## Candidates` markdown when `finalize` fired OR ≥3 candidates accumulated; existing briefing structure when `summarize` fired; defensive fallbacks otherwise.

**Self-eval + meta-eval** (`agent/evaluate.py`, `agent/meta_eval.py`, `agent/operator_sim.py`, `agent/replay.py`).
- Per-run critique: structure check + trace anti-patterns + Haiku judge.
- Hybrid canary autonomy: meta-eval proposes anchored edits to `SYSTEM_PROMPT`; each edit replays K cached-doc runs with the candidate prompt; a composite verdict (operator-coverage + structure + token efficiency + replay errors) decides promote / discard / gate.
- Output-shape aware metric dispatch: `goal-v1` records for candidates-list runs, `composite-v1` for briefings; both flow through the same demonstrations layer.
- See `docs/solutions/2026-05-27-meta-eval-strict-evaluation.md` for an honest assessment of where the canary works and where it doesn't.

**Compounding via demonstrations.** Once ≥3 composite-scored edit-history records exist, kept/reverted runs are surfaced as few-shot examples appended to `SYSTEM_PROMPT` on subsequent invocations. Pre-composite records are gated out to prevent teaching old-metric biases.

---

## Repo layout

```
agent/
  __main__.py         CLI entry
  loop.py             Tool-use loop, terminator, output-shape dispatch
  prompts.py          SYSTEM_PROMPT + SUMMARIZE_PROMPT
  llm.py              Anthropic SDK wrapper + retries
  storage.py          Flat-file persistence
  evaluate.py         Self-eval (structure check + trace critique + Haiku judge) + demonstrations
  operator_sim.py     coverage_score + goal_coverage_score + composite_score
  meta_eval.py        propose / apply / compare / keep / revert / status / score / autonomous
  replay.py           Cached-doc loop replay (canary foundation, offline iteration)
  tools/
    search.py         SearXNG client
    fetch.py          httpx + trafilatura, URL-hash cache
    summarize.py      Narrative briefing (shape B terminal tool)
    candidates.py     add_candidate + finalize (shape A tools)

docs/
  brainstorms/        Requirements docs
  plans/              Implementation plans
  solutions/          Compounded learnings + strict eval writeup
  context/            Reference inputs

tests/                Mocked unit + integration coverage
data/                 Generated artifacts (briefings, logs, fetched cache) — mostly gitignored
```

---

## CLI verbs

```bash
python -m agent "<goal>"                       # full agent run
python -m agent --no-eval "<goal>"             # skip self-eval pass

python -m agent.meta_eval propose [--no-llm]   # Sonnet-grade meta-eval over evals + edit history
python -m agent.meta_eval apply <edit_id> [--yes]
python -m agent.meta_eval compare "<prompt substring>"
python -m agent.meta_eval keep [--note "..."]
python -m agent.meta_eval revert [--note "..."]
python -m agent.meta_eval status               # dashboard: trailing edit + demonstrations readiness
python -m agent.meta_eval score <briefing>     # offline composite scoring of one briefing
python -m agent.meta_eval autonomous [--max-cycles N] [--max-cost N]
```

---

## Status

This is a portfolio piece + personal research tool. Not packaged for distribution. The persistence layer is flat files; the LLM provider is hardcoded to Anthropic; the search backend is a local SearXNG. The system intentionally stays single-process, single-prompt, single-domain-per-run.

Historical artifacts in `docs/plans/`, `docs/brainstorms/`, and `docs/solutions/` document how the system evolved — from a narrow take-home into the current generic shape — and what worked / what didn't under real LLM measurement.
