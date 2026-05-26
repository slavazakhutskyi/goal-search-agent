"""SearXNG-backed search tool. Spec-mandated (R2).

Returns a ranked list of {title, url, snippet}. Graceful on timeout, empty results,
and non-JSON responses — never raises to the agent loop (R13). Errors surface as a
dict with an `error` key instead.
"""

import os

import httpx

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
MAX_RESULTS = 10
TIMEOUT = 15.0


TOOL_SCHEMA = {
    "name": "search",
    "description": (
        "Search the web via a local SearXNG instance. "
        "Returns a ranked list of up to 10 results, each with title, url, snippet. "
        "Use focused queries — site:domain.com and date qualifiers ('past 7 days') help."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        "required": ["query"],
    },
}


def run(query: str) -> dict:
    """Call SearXNG /search?format=json. Loud failure → error dict, never raise."""
    try:
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "safesearch": 0},
            timeout=TIMEOUT,
            headers={"User-Agent": "rapidsos-intel-agent/0.1"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"SearXNG request failed: {exc!r}", "results": []}

    try:
        data = response.json()
    except ValueError as exc:
        return {"query": query, "error": f"SearXNG returned non-JSON: {exc}", "results": []}

    results = []
    for item in (data.get("results") or [])[:MAX_RESULTS]:
        results.append({
            "title": (item.get("title") or "").strip(),
            "url": item.get("url") or "",
            "snippet": (item.get("content") or "").strip(),
        })

    return {"query": query, "results": results}
