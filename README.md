# RapidSOS Competitive & Market Intelligence Agent

A CLI agent that takes one prompt, monitors the web via a local SearXNG instance, and returns a structured markdown briefing.

Built for the Senior Software Engineer II, AI Operations take-home (RapidSOS).

---

## What it does

Take one operator prompt. Plan focused web searches via SearXNG. Fetch the 4-8 most relevant sources. Produce one markdown briefing with TL;DR, themes, notable mentions, sentiment, and numbered citations. Save the briefing to `data/briefings/`. Log every run to `data/logs/runs.jsonl` for post-hoc analysis.

Designed for prompts like:

- *"Give me a briefing on everything published about RapidSOS in the last 7 days."*
- *"What are the top 3 public safety AI stories from this week?"*
- *"Find any press releases or news mentions of our competitors: Carbyne, RapidDeploy, Prepared."*
- *"Summarize the sentiment of recent coverage of AI in emergency dispatch."*

All four prompts route through the same loop — no per-prompt hardcoding.

---

## Quick start

```bash
# Prereqs: Docker, Python 3.11+, an Anthropic API key

docker compose up -d                          # boots SearXNG on :8888
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                          # then add ANTHROPIC_API_KEY

python -m agent "Give me a briefing on RapidSOS in the last 7 days"
```

The briefing prints to stdout. A markdown copy is saved to `data/briefings/`, and one line per run is appended to `data/logs/runs.jsonl`.

### Run the tests

```bash
pytest -v
```

All 38 tests pass on a fresh checkout without API key or running Docker — the test suite mocks the Anthropic client and SearXNG.

---

## Example output

See [`data/briefings/`](data/briefings/) for full real outputs from committed runs.

<!-- INLINE_EXAMPLE_BRIEFING_START -->

The 7-day RapidSOS prompt produced this briefing (model: Sonnet, source: `data/briefings/20260526-143055-...md`):

```markdown
# Briefing — Give me a briefing on everything published about RapidSOS in the last 7 days.
*Generated 2026-05-26T14:30:29+00:00 · 6 sources · sentiment: positive*

## TL;DR
RapidSOS announced a major partnership with AT&T on April 29, 2026, integrating HARMONY AI
directly into AT&T's ESInet infrastructure, eliminating the security tradeoff agencies
previously faced between trusted 911 networks and real-time intelligence over public
internet.[^1] The company hosted Innovation Day 2026 in Reno, Nevada, showcasing deployments
including North Carolina's statewide real-time interoperability system validated during
Hurricane Helene response.[^4] Kansas City PD became the first regional agency to deploy
RapidSOS translation capabilities supporting nearly 200 languages ahead of World Cup hosting
duties.[^6] Actor Jeremy Renner joined as partner and investor following his 2023 near-fatal
accident, with a documentary "Behind the Emergency" premiering at Innovation Day.[^2]

## Key themes
- **AT&T ESInet Integration**: HARMONY AI now runs directly on AT&T's ESInet infrastructure,
  the nation's most widely deployed NG911 network serving 88+ million people...
- **North Carolina Hurricane Helene Validation**: Following Hurricane Helene, North Carolina
  extended RapidSOS Real-Time Interoperability statewide after AT&T ESInet successfully
  rerouted calls up to 220 miles away...
- **Real-time Translation Deployment**: Kansas City PD launched RapidSOS translation
  supporting 147 languages via text-to-type and 50+ via automatic call translation...
- **AI Philosophy and Co-Pilot Model**: RapidSOS positioned HARMONY AI as "co-pilot, not
  replacement," explicitly acknowledging that "no AI is right 100% of the time"...
- **Jeremy Renner Partnership**: Actor Jeremy Renner became partner and investor after his
  January 2023 snowcat accident...

## Notable mentions
> "When Helene hit, AT&T ESInet™ kept our PSAPs connected amid destruction..."
> — L.V. Pokey Harris, Executive Director, North Carolina 911 Board[^1]

## Sentiment
**Positive** — Coverage emphasizes major infrastructure partnerships, real-world validation
during disaster response, state-level procurement pathways opening...

## Sources
[^1]: RapidSOS and AT&T Unite the World's Largest Safety Network... — prnewswire.com, 2026-04-29
[^2]: Jeremy Renner Partners with RapidSOS — finance.yahoo.com, 2026-04-15
[^4]: What RapidSOS Announced at Innovation Day 2026 — linkedin.com, 2026-05-07
[^6]: KCPD launches translation system for 911 calls ahead of World Cup — kctv5.com, 2026-04-13
[full briefing in data/briefings/]
```

**Note on the committed briefings.** `data/briefings/` contains three real runs:
the 7-day RapidSOS prompt above (Sonnet 4.6), plus two Haiku-generated briefings
(competitors, RapidSOS 7d v1) kept from cost-bounded smoke-test runs. The default
deployed model is Sonnet (`agent/llm.py:SONNET_MODEL`); the Haiku briefings are
preserved as honest artifacts demonstrating the loop runs cleanly on a cheaper model
when budget matters. Two of the four spec prompts (top-3-stories, sentiment) were
generated during smoke-test and not re-run on Sonnet — they're covered by the
behavior tests in `tests/test_acceptance.py` and trivially re-runnable.

<!-- INLINE_EXAMPLE_BRIEFING_END -->

---

## Design decisions and tradeoffs

### Python over TypeScript

My production agent stack at Aibility was TypeScript with LangGraph. I picked Python here because the AI Operations JD names LangChain / LlamaIndex as nice-to-have, the Anthropic SDK is Python-first, and pytest ergonomics are simpler than the JS test-runner config-soup for a CLI tool. I'm directing Claude Code at implementation time rather than hand-typing — what matters is read-fluency, not write-fluency.

### Direct Anthropic SDK over LangChain / LangGraph

The agent loop is ~150 lines in `agent/loop.py`, fully debuggable, no abstraction leak. For three tools and a single-turn CLI, a framework's overhead exceeds its value. LangGraph is the right call when I need persistent graph state, branching, or human-in-the-loop checkpoints — none of which this scope needs. I've shipped LangGraph in production; choosing *not* to use it here is the judgment signal.

### Single Claude Sonnet (no Haiku/Sonnet split)

One model = one mental model to defend. A two-model split adds wiring overhead and asks me to defend an unmeasured cost claim. For a 3-hour take-home that's discipline theater, not real value. Hardcoded in `agent/llm.py:SONNET_MODEL`.

### Markdown output, not JSON

The prompts in the spec are written in operator language ("briefing", "top 3 stories"). The consumer is a human reading this in Slack or email, not a downstream pipeline. JSON would be more machine-readable; markdown is more human-readable. A `--format json` flag is a 10-line live extension if needed.

### Behavior tests, not unit-coverage tests

Per spec prompt: one schema-shape test asserting the briefing has TL;DR, key themes (≥1), sources (≥1), and sentiment in `{positive, neutral, negative}`. Plus per-tool smoke tests (happy + error paths) — 38 tests total. The tests run fully mocked: no API key required, no Docker required, `pytest` works on a fresh clone in seconds. Real-API validation happens manually when generating committed briefings (the artifacts in `data/briefings/`).

### Flat-file storage + `runs.jsonl` observability

Spec mandates no DB. `data/fetched/{url-hash}.json` doubles as a within-run and cross-run cache. `data/briefings/{ts}-{slug}.md` holds outputs. `data/logs/runs.jsonl` is one line per run with prompt, status, iterations, elapsed time, and tool-call trace — greppable, post-hoc analyzable. Lightweight observability without an observability vendor; in production I'd graduate to Langfuse or OpenTelemetry on day one.

### Loud failure, graceful degradation

- **Network errors on fetch**: cached as error payload, agent continues with whatever else was fetched
- **LLM rate limit**: exponential backoff (3 retries) inside `llm.call_with_retry`
- **Non-retryable LLM error** (auth, bad request): logged to `runs.jsonl` with `status=partial_llm_error`, stub briefing written, CLI exits 1. No traceback in stdout.
- **Malformed HTML**: trafilatura JSON extract → fallback to raw text extract
- **Empty search results**: stub "no sources found" briefing rather than crash
- **Iteration cap (8) hit**: `status=partial_max_iterations`, reuses any successful summarize result rather than wasting another LLM call

For an operational intelligence tool, **partial signal beats no signal**. Silent swallow is the anti-pattern deliberately avoided.

### What I deliberately left out

- **Vector DB / RAG** — flat files sufficient at scope; defense available if asked
- **Multi-agent system** — single agent with 3 tools is enough; coordination cost without quality lift
- **Per-prompt-type routing** — all 4 spec prompts go through the same loop
- **Scheduled runs + Slack delivery** — where v2 operator value lives; first thing I'd build on day two
- **Cross-run dedup, source clustering, source-reliability scoring, sentiment trending** — matter at scale, not at 3-hour scope
- **Langfuse / observability vendor** — `runs.jsonl` is sufficient at this scope; production day-one add
- **Eval harness with labeled set** — for sentiment calibration via Cohen's kappa, briefing-quality scoring against held-out prompts
- **LLM-as-judge** — would add a second eval step requiring its own justification
- **Two-model Haiku/Sonnet split** — discipline theater without a measurement to defend
- **MCP server** — exposing the agent as a tool to Claude Code; nice for v2
- **Web UI** — out of spec
- **Deployment** — spec says local only

---

## Limitations and failure modes

- **Search quality depends on SearXNG instance health and engine availability.** Some engines (Google) get rate-limited from local Docker IPs; `searxng/settings.yml` enables four engines (Google, DuckDuckGo, Bing, Brave) to spread risk. On niche public-safety topics SearXNG may return weak results — in production I'd add a `--source` flag for known feeds (NENA, IWCE, RapidSOS blog, Axon investor relations) as a second-source path.
- **LLM may hallucinate dates or attribute quotes incorrectly.** Sentiment is a signal, not ground truth. Every claim has a numbered footnote so the operator can verify in one click; for production I'd add a "quote-not-found-in-source" guard that re-checks attributed quotes against the fetched body.
- **Publish dates depend on what trafilatura extracts** from page metadata. If absent, the briefing surfaces "date unknown" — never fabricated.
- **No content-freshness guarantee** beyond what SearXNG returns. Time-bounded prompts ("last 7 days") rely on the model adding date qualifiers and on SearXNG honoring them — neither is enforced.
- **Iteration cap = 8.** Chosen from first principles (3 phases × 2-3 calls). On prompts that exceed it, `status=partial_max_iterations` and the briefing is whatever the most-recent summarize call produced. Watch `runs.jsonl` after first runs — if any prompt routinely hits 8, raise the cap.
- **No CI.** Tests run locally via `pytest`. For production this would graduate to a hosted CI with the LIVE_API tests gated behind a paid Anthropic key.

---

## Architecture

### How a single run flows

```mermaid
flowchart TD
    CLI["python -m agent &quot;&lt;prompt&gt;&quot;"] --> Loop[agent/loop.py<br/>tool-call loop · cap=12]
    Loop -->|tool_use: search| Search[agent/tools/search.py]
    Loop -->|tool_use: fetch| Fetch[agent/tools/fetch.py]
    Loop -->|tool_use: summarize| Summarize[agent/tools/summarize.py]
    Loop -->|every iteration| Anthropic[(Anthropic API<br/>Sonnet 4.6)]

    Search --> SearXNG[(SearXNG<br/>Docker :8888)]
    Fetch --> WebPages[(Web pages<br/>via httpx + trafilatura)]
    Fetch --> Cache[(data/fetched/<br/>URL-hash JSON cache)]
    Summarize --> Anthropic

    Loop -->|status, iterations,<br/>tool calls, errors| RunsLog[(data/logs/runs.jsonl)]
    Loop -->|final markdown| Briefing[(data/briefings/<br/>{ts}-{slug}.md)]
    Loop --> Stdout[stdout: briefing<br/>stderr: status line]

    classDef external fill:#e8f0fe,stroke:#4285f4,color:#000
    classDef storage fill:#fef7e0,stroke:#fbbc04,color:#000
    classDef code fill:#e6f4ea,stroke:#34a853,color:#000
    class CLI,Loop,Search,Fetch,Summarize code
    class Anthropic,SearXNG,WebPages external
    class Cache,RunsLog,Briefing storage
```

### Why this shape

- **One file per concern** — `loop.py` orchestrates; each `tools/*.py` does one thing; `storage.py` handles all flat-file I/O; `llm.py` is the only Anthropic touch-point; `prompts.py` is the only place the LLM-facing contract lives.
- **Adding a tool = three small edits**: new file in `agent/tools/`, one entry in `TOOL_REGISTRY` in `loop.py`, one mention in `agent/prompts.py:SYSTEM_PROMPT`.
- **Tool registry uses module references**, not function refs — so tests can patch `agent.tools.<name>.run` and the loop picks up the patched version at call time.
- **Storage and tool layers don't know about each other** — `loop.py` is the only thing that imports both. Each tool returns a plain dict; storage receives plain dicts. No shared types crossing the boundary.

### Directory layout

```
agent/
  __main__.py     # CLI: python -m agent "<prompt>"
  loop.py         # tool-call loop, 12-iteration cap, partial-status fallbacks
  llm.py          # Anthropic SDK wrapper (single Sonnet), exponential-backoff retry
  prompts.py      # SYSTEM_PROMPT + SUMMARIZE_PROMPT
  storage.py      # flat-file I/O + URL-hash cache + runs.jsonl
  tools/
    search.py     # SearXNG client
    fetch.py      # URL → cleaned text + metadata via trafilatura, cached
    summarize.py  # docs → briefing markdown via single Sonnet call
eval/             # light eval harness — see "Evals" section below
tests/            # 38 fully-mocked tests; runs cold without credentials or Docker
data/
  briefings/      # real agent outputs (tracked)
  fetched/        # URL-hash cache (gitignored)
  logs/           # runs.jsonl + agent.log (gitignored)
docker-compose.yml + searxng/settings.yml
docs/             # brainstorm + plan (provenance for the work)
```

The code is intended to be re-readable by the author in 5 days — that's the bar.

---

## Run-time observability

Two complementary log streams, both flat-file (no vendor dependency at this scope):

**`data/logs/runs.jsonl`** — one JSON line per run. Machine-readable, greppable, plottable. Includes prompt, status, iterations, elapsed time, **input/output token totals**, per-tool latency, error fields, briefing path.

```jsonl
{"timestamp":"2026-05-26T14:30:14Z","prompt":"Give me a briefing on...","status":"complete","iterations":4,"elapsed_seconds":18.7,"tokens":{"input":47210,"output":3892},"tool_calls":[{"iteration":1,"name":"search","elapsed_seconds":0.8,"error":null},{"iteration":2,"name":"fetch","elapsed_seconds":2.1,"error":null},...],"briefing_path":"data/briefings/20260526-143014-give-me-a-briefing-on-rapidsos.md"}
```

**`data/logs/agent.log`** — human-readable streaming narration. What the agent is doing in real time. Useful for tailing during long runs or post-hoc debugging without parsing JSON.

```
[2026-05-26T14:30:14+00:00]  INFO agent run start  prompt='Give me a briefing on...' model=claude-sonnet-4-6
[2026-05-26T14:30:16+00:00]  INFO tool search  iteration=1 elapsed=0.8 error=
[2026-05-26T14:30:19+00:00]  INFO tool fetch  iteration=2 elapsed=2.1 error=
[2026-05-26T14:30:20+00:00]  WARN tool fetch  iteration=2 elapsed=20.1 error=fetch failed: ConnectError(...)
[2026-05-26T14:30:34+00:00]  INFO agent run end status=complete  iterations=4 elapsed=18.7 tokens_in=47210 tokens_out=3892 tools=6
```

Loud failure surfaces in both streams (`status=partial_llm_error` with `llm_error` field, `status=partial_max_iterations`, `status=partial_empty_response`). This is **the demonstrative pattern, not a production observability stack** — in production, `storage.log_event` would feed Langfuse / OpenTelemetry / Datadog with the same call signature.

---

## Evals

A light hybrid eval harness in `eval/` — deterministic schema checks + LLM-as-judge — to show I think about briefing quality measurement from day one, not as a v2 afterthought.

**Run:**

```bash
# Free (no API calls — schema-shape only)
python -m eval --no-judge data/briefings/

# With LLM judge (~$0.0005 per briefing on Haiku)
python -m eval data/briefings/
```

**What it checks:**

| Layer | Cost | What it scores | When to trust |
|---|---|---|---|
| **Deterministic** (`eval/checks.py:deterministic_check`) | $0 | Briefing has `# Briefing` header, `## TL;DR` section, `## Key themes`, `## Sources`, ≥1 footnote citation, sentiment in `{positive, neutral, negative}` | Always — same assertions as `tests/test_acceptance.py`, but at runtime against real briefings |
| **LLM judge** (`eval/checks.py:llm_judge`) | ~$0.0005/briefing | Haiku scores 1–5 on `coherence`, `citation_discipline`, `sentiment_fit` + one-sentence note on weakest axis | With caveats — Haiku judging Sonnet output has known bias; production would calibrate against a human-labeled set |

**Output:** per-file scorecard to stdout + machine-readable JSON to `data/logs/eval-latest.json`.

**What this deliberately is NOT:** a full eval framework. The production version (a labeled set of 30+ articles for sentiment calibration via Cohen's kappa, regression suite on a held-out prompt suite, judge-quality calibration, eval CI gate) is named in *What I deliberately left out* and is the day-2 build. This light harness exists so the pattern is in the repo from day one — extension is one new function in `checks.py`, not a new directory.

---

## Provenance

This repo includes the brainstorm + plan documents that drove the build:

- [`docs/brainstorms/2026-05-26-001-rapidsos-takehome-requirements.md`](docs/brainstorms/2026-05-26-001-rapidsos-takehome-requirements.md) — product requirements (problem framing, scope boundaries, key decisions D1–D9)
- [`docs/plans/2026-05-26-001-feat-rapidsos-intel-agent-plan.md`](docs/plans/2026-05-26-001-feat-rapidsos-intel-agent-plan.md) — implementation plan with U1–U6 units and acceptance coverage matrix

These show the brainstorm → plan → review → work pipeline this was built under. They're optional reading for evaluation — the artifact stands on its own.
