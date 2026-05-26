---
type: feat
status: active
created: 2026-05-26
origin: docs/brainstorms/2026-05-26-001-rapidsos-takehome-requirements.md
title: RapidSOS Competitive & Market Intelligence Agent
---

# feat: RapidSOS Competitive & Market Intelligence Agent

## Summary

Build a Python CLI agent that takes one prompt, runs a direct-Anthropic-SDK tool-call loop over `search` (SearXNG) → `fetch` → `summarize`, and outputs a markdown briefing. Five linear implementation units (U1-U4 + U6; U5 gap is intentional, see History note) delivering full brainstorm coverage with two test layers: per-tool smoke tests and per-prompt behavior tests using mocked Anthropic+SearXNG. Real-API validation happens organically in U6 when generating committed briefings. Target: ~4-5h build + finalize, ownership-first (every line defendable), no adjacent refactors.

**History:** Initial plan had 6 units with U5 as a separate real-API behavior-test unit. After review, U5 was folded into U4 with mocked tests (decision SG-02 in review round 1) — this eliminates `LIVE_API` infrastructure, removes API cost from tests, makes `pytest` runnable cold without credentials (directly supporting AE6), and shrinks the per-unit review surface. U5 ID is preserved as a gap per U-ID stability rule rather than renumbered.

## Problem Frame

This plan executes the take-home defined in `docs/brainstorms/2026-05-26-001-rapidsos-takehome-requirements.md` (see origin for full product framing). The artifact is a CLI agent that handles four spec prompts via the same loop and produces structured markdown briefings. The interview defense will be live (60 min, Martin Vejmelka + John Katt) with an unknown 20-30 min live extension via Claude Code.

The plan optimizes for **defensibility under live observation** over abstract code quality: minimal abstraction, boring patterns, explicit boundaries, behavior-level test coverage. Implementation is Claude-Code-driven; the candidate is architect-in-the-loop.

---

## Acceptance Coverage Matrix

Cross-reference from PDF take-home spec → brainstorm R/AE IDs → plan units → test artifacts. Every PDF criterion traces to a concrete deliverable.

| PDF Spec Criterion | Brainstorm | Plan Unit | Test / Artifact |
|---|---|---|---|
| Tool: `search` (SearXNG local Docker, ranked list of `{title, url, snippet}`) | R2 | U2 | `tests/test_search.py` |
| Tool: `fetch` (full text, HTML stripping, metadata stored: source_domain, publish_date, content_type) | R3 | U3 | `tests/test_fetch.py`, `tests/test_storage.py` |
| Tool: `summarize` (structured briefing with key themes, notable quotes/data points, sentiment) | R4 | U4 | `tests/test_summarize.py` |
| Python OR TypeScript | D1 (Python) | U1 | n/a — language locked in scaffold |
| LLM + agentic framework choice, defendable | D2, D3 (direct Anthropic SDK, single Sonnet) | U4 | n/a — defense in live conversation |
| Flat-file storage (no DB) | R6 | U1, U3 | `tests/test_storage.py` |
| CLI runnable with single prompt | R5 | U4 | `tests/test_cli.py` |
| Short README (how to run, what it does, design decisions/tradeoffs) | R7 | U6 | `README.md` |
| Sample prompt #1 — RapidSOS last 7 days | R1, AE1 | U4 | `tests/test_acceptance.py::test_prompt1_rapidsos_7day` |
| Sample prompt #2 — top 3 public safety AI stories | R1 (no dedicated origin AE) | U4 | `tests/test_acceptance.py::test_prompt2_top_3_stories` |
| Sample prompt #3 — competitor mentions (Carbyne, RapidDeploy, Prepared) | R1, AE2 | U4 | `tests/test_acceptance.py::test_prompt3_competitors_with_parents` |
| Sample prompt #4 — sentiment of AI in emergency dispatch | R1, AE3 | U4 | `tests/test_acceptance.py::test_prompt4_sentiment_3class` |
| Graceful failure (network errors, rate limits, malformed HTML, empty search) | R13, AE4 | U2, U3, U4 | error-path tests in each tool's test file |
| GitHub repo submission to `mcho@rapidsos.com` 24h+ before interview | R8 | U6 | manual process (handoff checklist) |
| Behavior tests pass green before submission | R10, AE5 | U4 | `pytest tests/test_acceptance.py` |
| Cold-run reproducibility (<2 min on fresh checkout) | AE6 | U6 | manual verification step + README "Quick start" |
| Eval axis: **product thinking** (output format, briefing actionable) | D4, D8 | U4 (markdown), U6 (README example) | n/a — surfaced in artifact + README |
| Eval axis: **engineering quality** (clean, modular, robust) | R9, R13 | All units | All tests + 2 review rounds |
| Eval axis: **AI fluency** (smart LLM use, tool calls, awareness of limits) | D3, R4, prompts file | U4 | `agent/prompts.py` + README "Design decisions" |
| Eval axis: **judgment** (what left out and why) | Scope Boundaries below | doc + README | README "What I deliberately left out" |
| Part 2 readiness: code readable in 5 days; live-extendable via Claude Code | R9 | All units (boring/readable) | manual: dry-run live extension morning of interview |
| Part 2 readiness: fluent AI tool use as positive signal | D8 | All units | n/a — workflow demonstration during live |

---

## Output Structure

Greenfield repo. Expected layout after U1-U6:

```text
RapidSOS/
├── agent/
│   ├── __init__.py
│   ├── __main__.py          # CLI: python -m agent "<prompt>"
│   ├── loop.py              # tool-call loop, max-iterations cap
│   ├── llm.py               # Anthropic SDK wrapper (single Sonnet)
│   ├── prompts.py           # SYSTEM_PROMPT, SUMMARIZE_PROMPT
│   ├── storage.py           # flat-file I/O + URL-hash cache + runs.jsonl
│   └── tools/
│       ├── __init__.py
│       ├── search.py        # SearXNG client
│       ├── fetch.py         # URL → text + metadata, trafilatura
│       └── summarize.py     # docs → briefing markdown via Sonnet
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # pytest fixtures: mock_anthropic_briefing, mock_search_results, mock_fetch_html
│   ├── test_search.py
│   ├── test_fetch.py
│   ├── test_storage.py
│   ├── test_summarize.py
│   ├── test_loop.py
│   ├── test_cli.py
│   └── test_acceptance.py   # 4 per-prompt schema-shape tests (mocked, no env-gating)
├── data/
│   ├── briefings/           # real agent outputs (1-2 committed)
│   ├── fetched/             # gitignored URL-hash cache
│   └── logs/                # gitignored runs.jsonl
├── searxng/
│   └── settings.yml
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docs/                    # brainstorm, plan, research inputs (kept in repo)
```

This is a scope declaration showing expected output shape; the per-unit `**Files:**` sections below are authoritative for what each unit creates.

---

## Key Technical Decisions

Carried forward verbatim from origin brainstorm (see origin: `docs/brainstorms/2026-05-26-001-rapidsos-takehome-requirements.md` §Key Decisions D1-D9). Restated here for plan-local quick reference:

- **D1 — Python over TypeScript.** Stronger AI-agent corpus, simpler pytest, Anthropic SDK Python-first. Live coding via Claude Code; candidate's role is architect-in-the-loop, not hand-typer.
- **D2 — Direct Anthropic SDK over LangGraph/LangChain.** 3 tools + single-turn = framework overhead > value. ~150-line loop, fully owned.
- **D3 — Single Claude Sonnet, no Haiku/Sonnet split.** One model = one mental model to defend. Cost-discipline theater explicitly rejected. (Exact model ID resolved at implementation time — see U4.)
- **D4 — Markdown output.** Operator-language prompts → human consumer. `--format json` flag is a 10-line live extension if asked.
- **D5 — Behavior tests over coverage tests.** Per-prompt schema validation = real contract; padding with unit coverage = theater.
- **D6 — Flat-file storage + `runs.jsonl` observability.** No vendor (Langfuse explicitly deferred). Greppable, post-hoc analyzable.
- **D7 — Iteration cap = 8, first-principles justification.** 3 phases × ~2-3 calls = 6-8 ceiling. Kill switch, not design target. NOT cited as Aibility data.
- **D8 — Defense from ownership, not scripts.** Hard live questions (Aimee transfer, AWS gap, domain humility) answered from understanding the code + honest calibration, not pre-rehearsed.
- **D9 — Scope = C with fallback ladder.** See Scope Boundaries below.

---

## Scope Boundaries

### In scope (this submission)

What U1-U6 deliver in this repo:

- All 3 mandated tools (`search`, `fetch`, `summarize`) per PDF spec
- Agent loop handling all 4 spec prompts via the same code path (no per-prompt branching)
- Behavior tests on each of the 4 prompts asserting briefing schema shape
- Per-tool smoke tests (happy + error paths)
- Graceful failure (one retry on fetch, exponential backoff on LLM rate limit, fallback HTML extraction, no-results handling)
- Flat-file storage: `data/fetched/{url_hash}.json` (cache + dedup), `data/briefings/{ts}-{slug}.md` (outputs), `data/logs/runs.jsonl` (observability)
- 1-2 real briefings committed in `data/briefings/`, honest about whatever SearXNG returns
- README per PDF: how to run, what it does, key design decisions/tradeoffs (3 sections, additionally listing "what I left out" as a tradeoffs sub-bullet)
- 2 code review rounds (correctness/simplicity → scope-guardian/adversarial) before submission
- Submission to GitHub + email to `mcho@rapidsos.com` 24h+ before interview

### Out of scope (next phases / v2)

Documented as planned future work, surfaced in conversation if asked "what would v2 look like":

- **Langfuse / observability vendor** — runs.jsonl is sufficient at 3h CLI scope; production day-1 add.
- **Eval harness with labeled set** — sentiment calibration via Cohen's kappa; briefing-quality scoring against a held-out prompt set.
- **Scheduled runs + Slack delivery** — Monday-morning briefing into `#intel`, on-demand via slash command. Where operator value compounds and the briefing becomes habit-forming.
- **Cross-run dedup + source clustering** — same press release shouldn't generate two themes across two runs.
- **Source reliability scoring** — domain trust weighting (TechCrunch vs SEO farm).
- **Sentiment trending over time** — "sentiment trended positive over 4 weeks", not just per-run snapshots.
- **Operator feedback loop** — thumbs-up/down per briefing → prompt tuning + source filter adjustment.
- **MCP server** — expose agent as a tool to Claude Code for internal Dev workflow.
- **AWS deployment shape** — Lambda runner, S3 briefing archive, EventBridge scheduler, Secrets Manager for keys, CloudWatch per-run cost metric.

### Universal anti-patterns excluded (not v2, just wrong)

- **RAG / vector DB** — flat files sufficient at volume; defense available.
- **Multi-agent architecture** — single agent with 3 tools is enough.
- **Per-prompt-type routing** — all 4 prompts route through same loop.
- **LLM-as-judge** — would add a second eval step requiring its own justification.
- **Two-model split (Haiku + Sonnet)** — explicitly rejected, see D3.
- **Pretty UI / web frontend** — out of PDF spec.
- **Deployment** — PDF says local only.

### Deferred to follow-up work (plan-local)

None. Plan is self-contained; no adjacent refactors pulled in (greenfield repo, nothing to drift into).

---

## System-Wide Impact

Greenfield repo — no existing systems affected. External dependencies introduced:

- **Docker** (SearXNG local instance on port 8888)
- **Anthropic API** (single key in `.env`, cost ~$0.02-0.10 per agent run depending on source count)
- **trafilatura** (Python HTML extraction library; well-maintained, MIT)
- **httpx** (async HTTP client for fetch + SearXNG)
- **pytest** (test framework)
- **anthropic** (official Python SDK)

No CI configured (out of scope — local-only per PDF). No deployment. No secrets in repo (gitignored `.env`).

---

## Implementation Units

Five units, linear dependency: U1 → U2 → U3 → U4 → U6 (U5 deleted per refactor; ID gap preserved per U-ID stability rule). Each unit lands as one atomic commit. Two review-round checkpoints noted in Execution Posture below.

### U1. Project scaffold + SearXNG verification

**Goal:** greenfield Python project ready to build on. SearXNG container running locally with `/search?format=json` endpoint verified end-to-end via curl. Env config in place.

**Requirements:** R5 (CLI shape), R6 (flat-file storage paths), R8 (repo setup)

**Dependencies:** none — foundation unit

**Files:**
- `requirements.txt`
- `docker-compose.yml`
- `searxng/settings.yml`
- `.env.example`
- `.gitignore`
- `agent/__init__.py` (empty)
- `agent/tools/__init__.py` (empty)
- `tests/__init__.py` (empty)
- `tests/conftest.py` (pytest fixtures stub; mocked-Anthropic + mocked-SearXNG/httpx fixtures will be added in U4)

**Approach:**
- `requirements.txt`: `anthropic>=0.40`, `httpx>=0.27`, `trafilatura>=1.12`, `python-dateutil>=2.9`, `pytest>=8.0`, `pytest-mock>=3.12`
- `docker-compose.yml`: SearXNG `latest` image, port 8888, mount `./searxng/settings.yml`
- `searxng/settings.yml`: required keys must be explicit (defaults will warn loudly or fail first-boot) — `server.secret_key` set to a non-default value (generate via `openssl rand -hex 32`), `server.limiter: false` (avoids Redis/Valkey sidecar dependency for the take-home scope), `search.formats: [html, json]` (json format is opt-in, not a top-level toggle), `search.safe_search: 0`. Enable 4 engines (Google, DuckDuckGo, Bing, Brave) to spread the rate-limit risk. First-boot pulls a 300-500MB image — budget realistically 45 min for U1, not 30, if no prior SearXNG experience
- `.gitignore`: `.env`, `__pycache__/`, `.venv/`, `data/fetched/`, `data/logs/` (briefings tracked)
- `tests/conftest.py`: bare pytest config now; fixtures (`mock_anthropic_briefing`, `mock_search_results`, `mock_fetch_html`) added in U4 as tests need them

**Patterns to follow:** standard Python `pyproject.toml`-or-`requirements.txt` Python layout; no clever monorepo or src/ structure.

**Test scenarios:**
- Test expectation: none — this is scaffold-only (no behavioral code). Verification is the curl smoke test below.

**Verification:**
- `docker compose up -d` returns clean
- `curl 'http://localhost:8888/search?q=rapidsos&format=json'` returns valid JSON with non-empty `results` array
- `pytest` runs (no tests yet, exits clean)
- `python -c "import anthropic, httpx, trafilatura"` imports succeed

---

### U2. `search` tool

**Goal:** SearXNG-backed search tool returning ranked list of `{title, url, snippet}`. Robust to timeout, empty results, non-200 responses.

**Requirements:** R2 (search tool per spec), R13 (graceful failure)

**Dependencies:** U1

**Files:**
- `agent/tools/search.py`
- `tests/test_search.py`

**Execution note:** Test-first for this unit — write the 4 test cases below, then make them pass. Search is the smallest, cleanest tool and the right place to establish the per-tool test rhythm.

**Approach:**
- Module-level `SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")`
- Constants: `MAX_RESULTS = 10`, `TIMEOUT = 15.0`
- Export `TOOL_SCHEMA` dict in Anthropic tool-use format: name, description (mention SearXNG, ranked results, focused queries help), input_schema with single `query: string` required field
- Function `run(query: str) -> dict` — call SearXNG `/search`, normalize to `{"query": ..., "results": [{"title", "url", "snippet"}]}`
- On `httpx.HTTPError` or `ValueError` (non-JSON): return `{"error": "...", "results": []}` — never raise to the loop

**Patterns to follow:** plain function (no class), explicit constants, return-dict-on-error pattern. Matches "boring + readable" mandate from D2/R9.

**Test scenarios:**
- `test_search_happy_path`: mock httpx GET returning 3 results, assert `run("rapidsos")` returns 3 normalized items with title/url/snippet keys
- `test_search_empty_results`: mock SearXNG returning `{"results": []}`, assert `run` returns `{"query": "...", "results": []}` (not an error)
- `test_search_http_error`: mock httpx raising `httpx.TimeoutException`, assert `run` returns dict with `error` key and empty `results`
- `test_search_non_json_response`: mock SearXNG returning text/html, assert `run` returns dict with `error` key

**Verification:**
- All 4 tests pass: `pytest tests/test_search.py -v`
- Manual smoke: `python -c "from agent.tools.search import run; print(run('rapidsos public safety'))"` returns real results (requires SearXNG container running)

---

### U3. `fetch` tool + storage with URL-hash cache

**Goal:** Fetch URL → clean text + metadata (source_domain, publish_date, content_type). Cache by URL hash in `data/fetched/`. Fallback to raw extraction on trafilatura failure. Graceful on network errors.

**Requirements:** R3 (fetch with metadata), R6 (flat-file storage), R13 (graceful failure)

**Dependencies:** U1

**Files:**
- `agent/tools/fetch.py`
- `agent/storage.py`
- `tests/test_fetch.py`
- `tests/test_storage.py`

**Execution note:** Test-first for `agent/storage.py` (pure function, easy to TDD). Fetch tool can be test-after-implementation since its main complexity is integration with httpx and trafilatura.

**Approach:**
- `agent/storage.py`: pure functions — `url_hash(url) -> str` (SHA-256 first 16 hex chars), `fetched_path(url) -> Path`, `save_fetched(url, payload)`, `load_fetched(url) -> dict | None`, `slugify(text) -> str`, `save_briefing(prompt, markdown) -> Path`, `log_run(entry: dict)` (JSONL append)
- Module-level `DATA / FETCHED / BRIEFINGS / LOGS` Path constants resolved from `Path(__file__).resolve().parents[1]`; `.mkdir(parents=True, exist_ok=True)` on import
- `agent/tools/fetch.py`: export `TOOL_SCHEMA`, `run(url: str) -> dict`
- `run` checks cache first → returns cached payload with `_cache_hit: True` marker
- Otherwise: `httpx.get(url, follow_redirects=True, timeout=20)`, extract via `trafilatura.extract(html, output_format="json", with_metadata=True)`, fallback to raw extract on JSON-parse failure
- Cap extracted text at 30,000 chars to bound summarizer context

**Patterns to follow:** same return-dict-on-error pattern as U2. Storage as pure utility module imported by tools and loop.

**Test scenarios for `tests/test_storage.py`:**
- `test_url_hash_deterministic`: same URL → same hash, two URLs → different hashes
- `test_url_hash_length`: hash is 16 hex chars
- `test_save_load_fetched_roundtrip`: save dict, load returns equivalent dict
- `test_load_fetched_missing_returns_none`: unsaved URL → `None`
- `test_slugify_normalization`: `"Give me a briefing!"` → `"give-me-a-briefing"`, length capped
- `test_save_briefing_creates_file_with_timestamp_slug`: returns Path; filename includes timestamp + slug
- `test_log_run_appends_jsonl`: two log_run calls → 2 lines in runs.jsonl, each valid JSON

**Test scenarios for `tests/test_fetch.py`:**
- `test_fetch_happy_path`: mock httpx + trafilatura returning extracted JSON with text/title/date → returns clean dict with metadata.publish_date populated
- `test_fetch_html_stripping_fallback`: trafilatura JSON extraction fails (returns None), raw extract succeeds → text populated, metadata mostly empty but no crash
- `test_fetch_cache_hit`: pre-save a fetched payload, call `run(url)` → returns cached with `_cache_hit: True`, no httpx call made
- `test_fetch_network_error`: mock httpx raising error → returns dict with `error`, empty text, cached so subsequent call is fast
- `test_fetch_caps_long_content`: extract returns 50K-char text → stored text is 30K chars, char_count metadata accurate
- Covers AE4 (graceful empty/error handling)

**Verification:**
- `pytest tests/test_fetch.py tests/test_storage.py -v` all pass
- Manual smoke: `python -c "from agent.tools.fetch import run; print(run('https://rapidsos.com/blog/').get('metadata'))"` returns dict with source_domain populated

---

### U4. Agent loop + summarize + CLI + observability + behavior tests

**Goal:** End-to-end working agent (mocked tests level). CLI takes one prompt, agent runs Anthropic tool-call loop with 8-iteration cap, summarize produces markdown briefing, runs.jsonl logs the run with status/iterations/elapsed/tool calls. Four behavior tests per spec prompt assert briefing shape against mocked Anthropic + mocked SearXNG/httpx (formerly U5, now folded in per SG-02 review decision).

**Requirements:** R1 (4 prompts via same loop), R4 (summarize w/ themes/quotes/sentiment), R5 (CLI single prompt), R10 (behavior tests on output contract), R13 (rate-limit backoff, iteration-cap partial briefing), AE1, AE2 (parent-company surfacing), AE3 (3-class sentiment), AE5 (tests green)

**Dependencies:** U1, U2, U3

**Files:**
- `agent/llm.py`
- `agent/prompts.py`
- `agent/tools/summarize.py`
- `agent/loop.py`
- `agent/__main__.py`
- `tests/test_summarize.py`
- `tests/test_loop.py`
- `tests/test_cli.py`
- `tests/test_acceptance.py` (4 schema-shape tests per spec prompt, fully mocked — no LIVE_API gating)
- `tests/conftest.py` (add `mock_anthropic_briefing`, `mock_search_results`, `mock_fetch_html` fixtures)

**Execution note:** Test-first for `summarize.run()` and `loop.run()` (mocked Anthropic client). CLI is thin — test the parser, mock loop.run. Behavior tests in `test_acceptance.py` use a fixture that returns canned briefing markdown matching the prompt theme; agent loop wires through normally with mocked tool returns.

**Technical design (directional, not implementation specification):**

```
loop.run(user_prompt):
    messages = [{role: user, content: user_prompt}]
    for iteration in 1..MAX:
        response = anthropic.messages.create(
            model=SONNET, system=SYSTEM_PROMPT,
            tools=[search.TOOL_SCHEMA, fetch.TOOL_SCHEMA, summarize.TOOL_SCHEMA],
            messages=messages, max_tokens=2048)
        messages.append({role: assistant, content: response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break  # final answer reached
        results = [dispatch(tu.name, tu.input) for tu in tool_uses]
        messages.append({role: user, content: [tool_result blocks]})
    extract briefing from last summarize() call OR fall back to text content
    storage.save_briefing(prompt, briefing)
    storage.log_run({prompt, status, iterations, elapsed, tool_calls, briefing_path})
    return {briefing, briefing_path, status, iterations, elapsed, tool_call_count}
```

This is directional only. Implementer uses Anthropic SDK actual response shape and Claude Code's help for exact serialization of tool_result content blocks.

**Approach:**
- `agent/llm.py`: lazy singleton `client()` returning `anthropic.Anthropic()`, constants `SONNET_MODEL = "claude-sonnet-4-5"` (verify exact ID at implementation time), helper `call_with_retry(create_kwargs, max_retries=3)` with exponential backoff on `RateLimitError`/`APIConnectionError`/`InternalServerError`
- `agent/prompts.py`: `SYSTEM_PROMPT` (one-paragraph agent instructions — plan 1-3 focused queries, fetch 2-4 results per search, stop fetching at 4-8 distinct sources, call summarize once at end, surface parent companies on competitor prompts per AE2), `SUMMARIZE_PROMPT` template with placeholders for prompt/documents/timestamp/n_sources, defines the 5-section markdown shape (TL;DR / Key themes with [^N] citations / Notable mentions / Sentiment with 3-class verdict / Sources footnotes)
- `agent/tools/summarize.py`: `TOOL_SCHEMA` declaring inputs (prompt, documents array), `run(prompt, documents)` formats documents into labeled blocks (`[Document N]\nURL: ...\nTitle: ...\nDate: ...\nText:\n...`) and calls Sonnet with `SUMMARIZE_PROMPT`. Returns `{"briefing": str}`. Handles empty-documents case with a "no sources found" stub briefing.
- `agent/loop.py`: `TOOL_REGISTRY = {"search": (schema, run), "fetch": (...), "summarize": (...)}`, `MAX_ITERATIONS = 8`, `run(user_prompt)` orchestrates tool-call loop. On cap hit: status = `"partial_max_iterations"`, fall back to calling summarize directly with whatever was fetched. Persists briefing + appends runs.jsonl entry.
- `agent/__main__.py`: `argparse` → `prompt` positional, call `loop.run(prompt)`, print briefing to stdout + status line to stderr, exit code 0 on complete / 1 on partial.

**Patterns to follow:** module-level functions over classes. Tool registry as dict (extension = one new entry). Explicit `TOOL_REGISTRY` declaration so Part 2 extensions are obvious.

**Test scenarios for `tests/test_summarize.py`:**
- `test_summarize_with_documents`: mock Anthropic client returning text block with markdown briefing, assert returned `briefing` contains TL;DR, Key themes, Sentiment, Sources sections
- `test_summarize_empty_documents`: pass empty list, assert returned briefing has TL;DR explaining no sources found, no Anthropic API call made
- `test_summarize_formats_documents_with_labels`: pass 2 documents, assert the prompt sent to Anthropic includes `[Document 1]` and `[Document 2]` labels with URL/title/date fields rendered

**Test scenarios for `tests/test_loop.py`:**
- `test_loop_natural_termination`: mock Anthropic returning tool_use for search → tool_use for summarize → final text. Assert loop runs 3 iterations, status `"complete"`, briefing extracted from summarize result
- `test_loop_max_iterations_partial`: mock Anthropic always returning a tool_use (never terminating). Assert loop stops at MAX_ITERATIONS, status `"partial_max_iterations"`, briefing falls back to summarize call with whatever was fetched
- `test_loop_unknown_tool`: mock Anthropic returning tool_use with name "unknown_tool", assert loop receives error tool_result and continues (doesn't crash)
- `test_loop_logs_run_to_jsonl`: assert after `run()`, runs.jsonl has new line with prompt, status, iterations, tool_calls fields
- `test_loop_persists_briefing_to_data`: assert after `run()`, a file exists in `data/briefings/` with the expected timestamp+slug name

**Test scenarios for `tests/test_cli.py`:**
- `test_cli_with_prompt_arg`: mock loop.run returning known briefing dict, assert stdout contains briefing, exit code 0
- `test_cli_no_args_returns_usage_error`: invoke with no args, assert stderr contains "usage:" and exit code 2
- `test_cli_partial_status_exit_1`: mock loop.run returning `status="partial_max_iterations"`, assert exit code 1

**Test scenarios for `tests/test_acceptance.py`** (formerly U5 — fully mocked, no LIVE_API gating):

Helper: `assert_valid_briefing(markdown: str, expected_sentiment_in: set[str] | None = None)` that checks `# Briefing` header present, `## TL;DR` non-empty, `## Key themes` has ≥1 bullet, `## Sources` has ≥1 footnote, sentiment line includes one of `{positive, neutral, negative}`.

Mock approach: `mock_anthropic_briefing` fixture returns a canned briefing matching the prompt theme; `mock_search_results` + `mock_fetch_html` feed canned tool returns via httpx mocking. Agent loop wires through normally. Tests assert SCHEMA only — NEVER content (e.g., never "must mention AT&T launch"); content correctness is U6's manual concern with real briefings.

- `test_prompt1_rapidsos_7day`: prompt = `"Give me a briefing on everything published about RapidSOS in the last 7 days."` → mocked briefing fed through loop, `assert_valid_briefing`. **Covers AE1.**
- `test_prompt2_top_3_stories`: prompt = `"What are the top 3 public safety AI stories from this week?"` → `assert_valid_briefing`. (No dedicated origin AE — covers R1 generic.)
- `test_prompt3_competitors_with_parents`: prompt = `"Find any press releases or news mentions of our competitors: Carbyne, RapidDeploy, Prepared."` → `assert_valid_briefing` AND assert briefing text contains at least one of `"Carbyne"`, `"Prepared"`, `"RapidDeploy"` AND at least one of `"Axon"`, `"Motorola"` (parent-company surfacing). **Covers AE2.**
- `test_prompt4_sentiment_3class`: prompt = `"Summarize the sentiment of recent coverage of AI in emergency dispatch."` → `assert_valid_briefing` with sentiment ∈ enum. **Covers AE3.**

AE5 ("behavior tests pass green") is satisfied implicitly by `pytest tests/test_acceptance.py` returning exit code 0.

**Verification:**
- `pytest tests/test_summarize.py tests/test_loop.py tests/test_cli.py tests/test_acceptance.py -v` all pass (no env vars required — fully mocked)
- `pytest` from a cold clone (after `pip install`) passes without API key
- Manual end-to-end smoke (NOT a test, real-API validation): `python -m agent "Test prompt"` runs against real Anthropic + SearXNG, produces a briefing in `data/briefings/`, appends a line to `data/logs/runs.jsonl`. This is the bridge into U6.
- **Iteration-cap measurement step (during manual smoke and U6 briefing runs):** `grep partial_max_iterations data/logs/runs.jsonl` → if any line matches, EITHER raise `MAX_ITERATIONS` to 12 and re-run the offending prompt, OR document the partial behavior in README §Limitations before U6 commits briefings. Prevents U6 from inlining a partial-status briefing as the headline artifact.

---

### U6. Real briefings + README + submission

**Goal:** 1-2 real briefings committed in `data/briefings/` (uncurated, honest artifacts only), README per PDF spec with one real briefing inlined as Example output, repo pushed to GitHub, email sent to `mcho@rapidsos.com`.

**Requirements:** R7 (README), R8 (GitHub submission), R12 (1-2 committed briefings), AE6 (cold-run < 2 min)

**Dependencies:** U1, U2, U3, U4

**Execution note:** Run the agent against each spec prompt. Commit the strongest 1-2 briefing outputs as-is (no editing for "polish" — honest artifacts only, per D5 and brainstorm §"If real briefings disappoint"). README written AFTER briefings exist so the inlined example is real, not templated.

**Files:**
- `data/briefings/<timestamp>-<slug>.md` × 1-2 (real outputs)
- `README.md`

**Approach:**

README structure (PDF says "short" → 3 main sections + 1 design-decisions tradeoffs sub-bullet for "what I left out"):

```markdown
# RapidSOS Competitive & Market Intelligence Agent

## What it does
[3-4 sentences plain English]

## Quick start
[Docker compose up, pip install, .env setup, python -m agent "<prompt>"]
[+ one full example invocation]

## Design decisions and tradeoffs
- Python + direct Anthropic SDK + single Sonnet — why (cites D1-D3)
- Markdown output — why (cites D4)
- Behavior tests + flat-file storage + runs.jsonl — why (cites D5-D6)
- Iteration cap at 8 — first-principles justification (cites D7)
- Graceful failure pattern — loud-log-degrade-gracefully (cites R13)
- What I deliberately left out and why — list from Scope Boundaries "Out of scope"

## Example output
[Inline the strongest of the 2 committed briefings]
```

Submission steps (manual; not part of the artifact):
1. `git init`, `git add -A`, commit
2. Create public GitHub repo (or private + add `mcho@rapidsos.com`)
3. `git push -u origin main`
4. Cold-run verification (AE6) — **must run from a fresh directory, NOT the dev dir** (avoids cached Docker image, populated `.env`, warm fetch cache, existing `.venv`). Recipe: `cd /tmp && git clone <repo-url> rapidsos-coldrun && cd rapidsos-coldrun && time (docker compose up -d && python -m venv .venv && .venv/bin/pip install -r requirements.txt && cp .env.example .env)` → manually add API key to `.env` → `time .venv/bin/python -m agent "Give me a briefing on RapidSOS in the last 7 days"` → total wall-clock should be under 2 min (excluding the manual `.env` edit). Without the fresh dir, AE6 is a claim, not a verification.
5. Email `mcho@rapidsos.com` with repo link 24h+ before interview

**Test scenarios:**
- Test expectation: none — this unit produces non-code deliverables (briefings, README, submission). Verification is manual.

**Verification:**
- 1-2 briefing files exist in `data/briefings/` with realistic content (no hand-editing)
- README sections complete; example briefing renders cleanly on GitHub
- Cold-run from fresh checkout works in < 2 minutes (manual stopwatch)
- Repo URL emailed to `mcho@rapidsos.com` 24h+ before interview slot

---

## Execution Posture

- **TDD-flavored, not pure red-green-refactor.** Per-unit test rhythm: write the test(s), run red, implement, run green, commit.
- **Test-first units:** U2 (search), U3 (storage portion), U4 (summarize + loop). Test-after-implementation acceptable for: U1 (scaffold, no tests), U3 (fetch tool, integration-heavy), U6 (manual deliverables).
- **Two code review rounds** insert between U4→U6 and after U6 code (before GitHub push):
  - **Round 1 (between U4 and U6):** `/ce-code-review` with reviewers `correctness` + `simplicity` + `maintainability`. Address P0/P1, defer P2/P3 with explicit note.
  - **Round 2 (after U6 code complete, before GitHub push):** `/ce-code-review` with reviewers `scope-guardian` + `adversarial`. Last-mile sanity check. Address P0 only — by this point everything else is deferred to v2 or accepted.
- **All tests are mocked-by-default** — `pytest` runs cleanly from a cold clone without any API keys or running Docker (supports AE6). Real-API end-to-end validation happens manually in U6 when generating committed briefings.
- **Fallback ladder** active under time pressure (from origin §D9): cut Round 2 review → cut 2nd briefing → drop one prompt's acceptance test if mock fixture work is too slow → hard floor (3 tools work + 1 real briefing committed + 3 of 4 behavior tests green + README).

---

## Risks & Dependencies

- **SearXNG result quality on niche public-safety topics is unverified.** Mitigation: verify in U1 with curl against `"rapidsos public safety"`. If results are weak, document in README "Limitations and failure modes" and proceed — do NOT hand-curate briefings to compensate.
- **Anthropic SDK tool-format details may shift.** Mitigation: use Claude Code at implementation time to handle exact response shape. Plan describes the intent (text block + tool_use block extraction); implementation handles the SDK reality.
- **trafilatura extraction can return None on aggressive paywalls / JS-heavy sites.** Mitigation: fallback to raw text extraction (R13). If both fail, the fetch tool returns an empty text payload + error metadata — agent loop continues with whatever else was fetched.
- **Anthropic rate limits during U6 briefing generation.** Mitigation: exponential backoff already designed (D7, llm.py `call_with_retry`). If still throttled, space briefing runs by 15 min; if some prompts return thin results, document in README Limitations rather than hand-curating.
- **Iteration cap = 8 might be too low for sentiment prompt** if model searches more aggressively. Mitigation: U4 verification step measures actual iteration count via runs.jsonl after the manual smoke run; if any prompt hits 8 routinely, raise to 12 OR surface as a finding in README Limitations before U6 commits briefings.

---

## Documentation Plan

- **README** is the user-facing artifact (U6).
- **`docs/brainstorms/2026-05-26-001-rapidsos-takehome-requirements.md`** stays in repo as scope/decision provenance (linked from this plan; visible to interviewers if they look).
- **`docs/plans/2026-05-26-001-feat-rapidsos-intel-agent-plan.md`** (this file) stays in repo as execution provenance.
- **`docs/99-research-inputs.md`** stays in repo as research bibliography — not pre-rehearsed defense scripts.
- **`runs.jsonl`** is gitignored — it's local observability, not a deliverable.

---

## Next step

Run `/ce-work` against this plan to execute U1 → U2 → U3 → U4 → U6 in sequence (U5 ID gap is intentional).
