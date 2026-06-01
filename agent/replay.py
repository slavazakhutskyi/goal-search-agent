"""Cached-doc replay layer. Foundation for the canary mechanism.

`replay_run(prompt, system_override, allowed_urls)` runs the full agent loop
against an alternate SYSTEM_PROMPT, but constrains the model to only see URLs
already cached in `data/fetched/`. The search tool is mocked to return ONLY
the allowed URLs as ranked results; the fetch tool already reads from cache
transparently when available. The result is a fully-executed loop run that
costs only the Sonnet calls — no SearXNG hits, no httpx fetches, no network.

This unlocks:
- Canary testing (compare two SYSTEM_PROMPTs against the same input space)
- Offline prompt iteration (~$0.03-0.10 per replay vs ~$0.05-0.15 for a live run)
- Regression testing (replay historical prompts after code changes)
- A/B model comparison (swap model IDs, replay same prompt)

The replay is DETERMINISTIC in inputs but not in outputs — LLM stochasticity
still applies. K=1 replay is a single-canary signal; K≥3 needed for
statistical claims. See plan U7 for the canary policy that consumes this.
"""

from contextlib import contextmanager

from agent import loop, prompts, storage
from agent.tools import fetch, search


def _build_mock_search(allowed_urls: list[str]):
    """Return a search.run replacement that yields ONLY the allowed URLs.

    Each "result" is composed from the cached doc's metadata + first 200 chars
    of text as snippet. The model sees a ranked list it can fetch from, but
    cannot wander outside the cached set.
    """
    docs = []
    for url in allowed_urls:
        cached = storage.load_fetched(url)
        if cached and cached.get("text") and not cached.get("error"):
            docs.append(cached)

    def mock_search(query: str) -> dict:
        results = []
        for d in docs:
            meta = d.get("metadata") or {}
            results.append({
                "title": meta.get("title") or "(no title)",
                "url": d.get("url") or "",
                "snippet": (d.get("text") or "")[:200],
            })
        return {"query": query, "results": results}

    return mock_search


def _passthrough_fetch():
    """fetch.run already reads from cache when present. We add a safety net:
    refuse any URL not in the allowed set so the model can't accidentally
    bypass the canary boundary."""

    def make(allowed_set):
        def mock_fetch(url: str) -> dict:
            if url not in allowed_set:
                return {"url": url, "error": "replay: URL not in canary cache", "text": ""}
            cached = storage.load_fetched(url)
            if cached:
                return {**cached, "_cache_hit": True}
            return {"url": url, "error": "replay: cache miss", "text": ""}
        return mock_fetch

    return make


@contextmanager
def cached_mode(allowed_urls: list[str], system_override: str | None):
    """Context manager: swap search.run + fetch.run + prompts.SYSTEM_PROMPT.

    Restores all three on exit, even on exception. Reentrant-unsafe — do not
    nest. Tests assert the originals are restored after the block exits.
    """
    original_search_run = search.run
    original_fetch_run = fetch.run
    original_system_prompt = prompts.SYSTEM_PROMPT

    search.run = _build_mock_search(allowed_urls)
    fetch.run = _passthrough_fetch()(set(allowed_urls))
    if system_override is not None:
        prompts.SYSTEM_PROMPT = system_override

    try:
        yield
    finally:
        search.run = original_search_run
        fetch.run = original_fetch_run
        prompts.SYSTEM_PROMPT = original_system_prompt


def replay_run(prompt: str, allowed_urls: list[str], system_override: str | None = None) -> dict:
    """Full-loop replay against a constrained URL set + optional alternate prompt.

    Args:
        prompt: the user prompt to run
        allowed_urls: URLs whose cached docs the model is allowed to use
        system_override: alternate SYSTEM_PROMPT text; None = use current

    Returns the dict from loop.run (briefing, status, iterations, tokens, etc.).
    `self_eval=False` always — replays don't generate eval records (that would
    pollute the lessons buffer with non-canonical runs).
    """
    with cached_mode(allowed_urls, system_override):
        return loop.run(prompt, self_eval=False)
