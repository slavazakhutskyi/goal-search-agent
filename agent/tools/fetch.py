"""Fetch tool. Spec-mandated (R3).

URL → cleaned text + metadata (source_domain, publish_date, content_type). Caches
per URL hash in data/fetched/ so re-fetches within and across runs are free.
Falls back to raw trafilatura extract on JSON-extraction failure. Network errors
return a payload with `error` key — never raise (R13).
"""

import json
from urllib.parse import urlparse

import httpx
import trafilatura

from agent import storage

TIMEOUT = 20.0
MAX_CONTENT_CHARS = 30_000  # cap to keep summarizer context bounded


TOOL_SCHEMA = {
    "name": "fetch",
    "description": (
        "Fetch the full text of a URL. Returns cleaned text plus metadata "
        "(source domain, publish date if available, content type). "
        "Cached per URL — re-fetching the same URL is free."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
        },
        "required": ["url"],
    },
}


def _empty_metadata(url: str, content_type: str = "") -> dict:
    return {
        "source_domain": urlparse(url).netloc,
        "content_type": content_type,
        "publish_date": None,
        "title": None,
        "author": None,
    }


def run(url: str) -> dict:
    """Fetch URL → {url, text, metadata, char_count}. Cached + graceful on error."""
    cached = storage.load_fetched(url)
    if cached is not None:
        return {**cached, "_cache_hit": True}

    try:
        response = httpx.get(
            url,
            timeout=TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "rapidsos-intel-agent/0.1"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # Do NOT cache errors — a transient 503 or timeout would poison the URL
        # forever. Subsequent runs should retry on their own.
        return {
            "url": url,
            "text": "",
            "metadata": _empty_metadata(url),
            "char_count": 0,
            "error": f"fetch failed: {exc!r}",
        }

    html = response.text
    content_type = response.headers.get("content-type", "").split(";")[0].strip()
    metadata = _empty_metadata(url, content_type=content_type)
    text = ""

    extracted = trafilatura.extract(
        html, output_format="json", with_metadata=True, include_comments=False
    )
    if extracted:
        try:
            data = json.loads(extracted)
            text = (data.get("text") or "")
            metadata["publish_date"] = data.get("date")
            metadata["title"] = data.get("title")
            metadata["author"] = data.get("author")
        except ValueError:
            text = ""

    if not text:
        # Fallback: raw text extraction without metadata
        text = trafilatura.extract(html, include_comments=False) or ""

    text = text[:MAX_CONTENT_CHARS]

    payload = {
        "url": url,
        "text": text,
        "metadata": metadata,
        "char_count": len(text),
    }
    storage.save_fetched(url, payload)
    return payload
