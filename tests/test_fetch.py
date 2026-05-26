"""Tests for agent.tools.fetch. Mocks httpx + trafilatura — no live HTTP."""

import json

import httpx

from agent import storage
from agent.tools import fetch


def _mock_httpx_response(mocker, *, text="", status=200, headers=None, raise_exc=None):
    if raise_exc is not None:
        mocker.patch("agent.tools.fetch.httpx.get", side_effect=raise_exc)
        return
    response = mocker.MagicMock(spec=httpx.Response)
    response.text = text
    response.status_code = status
    response.headers = headers or {"content-type": "text/html"}
    response.raise_for_status = mocker.MagicMock(return_value=None)
    mocker.patch("agent.tools.fetch.httpx.get", return_value=response)


def test_fetch_happy_path(mocker, temp_data_dir):
    """trafilatura returns JSON with text/title/date → metadata populated."""
    _mock_httpx_response(mocker, text="<html>...</html>")
    mocker.patch(
        "agent.tools.fetch.trafilatura.extract",
        return_value=json.dumps({
            "text": "Article body here.",
            "title": "Big News",
            "date": "2026-05-20",
            "author": "Jane Doe",
        }),
    )

    out = fetch.run("https://example.com/news")

    assert "error" not in out
    assert out["text"] == "Article body here."
    assert out["metadata"]["source_domain"] == "example.com"
    assert out["metadata"]["title"] == "Big News"
    assert out["metadata"]["publish_date"] == "2026-05-20"
    assert out["metadata"]["author"] == "Jane Doe"
    assert out["char_count"] == len("Article body here.")


def test_fetch_html_stripping_fallback(mocker, temp_data_dir):
    """trafilatura JSON extract returns None → fall back to raw extract; no crash."""
    _mock_httpx_response(mocker, text="<html><body><p>raw body</p></body></html>")
    # First call (JSON format) returns None, second call (raw) returns text
    mocker.patch(
        "agent.tools.fetch.trafilatura.extract",
        side_effect=[None, "raw body"],
    )

    out = fetch.run("https://example.com/fallback")

    assert "error" not in out
    assert out["text"] == "raw body"
    assert out["metadata"]["title"] is None
    assert out["metadata"]["publish_date"] is None
    assert out["metadata"]["source_domain"] == "example.com"


def test_fetch_cache_hit(mocker, temp_data_dir):
    """Pre-saved payload → returned with _cache_hit: True; no httpx call."""
    url = "https://example.com/cached"
    cached_payload = {
        "url": url,
        "text": "cached body",
        "metadata": {"source_domain": "example.com", "title": "cached"},
        "char_count": 11,
    }
    storage.save_fetched(url, cached_payload)
    httpx_get = mocker.patch("agent.tools.fetch.httpx.get")

    out = fetch.run(url)

    assert out["_cache_hit"] is True
    assert out["text"] == "cached body"
    httpx_get.assert_not_called()


def test_fetch_network_error(mocker, temp_data_dir):
    """httpx raising → returned payload has error key + empty text. Error NOT cached
    (transient errors must retry on subsequent calls — Round 2 reliability finding rel-001)."""
    _mock_httpx_response(mocker, raise_exc=httpx.ConnectError("dns failure"))

    out = fetch.run("https://nope.example.com/fail")

    assert "error" in out
    assert out["text"] == ""
    assert out["char_count"] == 0
    assert out["metadata"]["source_domain"] == "nope.example.com"
    # Crucially: error payload is NOT persisted to the cache.
    assert storage.load_fetched("https://nope.example.com/fail") is None
    # Second call retries httpx (no sticky cache from the first failure).
    httpx_get_spy = mocker.patch(
        "agent.tools.fetch.httpx.get", side_effect=httpx.ConnectError("dns failure"),
    )
    out2 = fetch.run("https://nope.example.com/fail")
    assert "error" in out2
    assert "_cache_hit" not in out2
    httpx_get_spy.assert_called_once()


def test_fetch_caps_long_content(mocker, temp_data_dir):
    """Extract returns 50K chars → stored text is 30K (MAX_CONTENT_CHARS); char_count matches."""
    long_text = "x" * 50_000
    _mock_httpx_response(mocker, text="<html>...</html>")
    mocker.patch(
        "agent.tools.fetch.trafilatura.extract",
        return_value=json.dumps({"text": long_text, "title": "Long", "date": None, "author": None}),
    )

    out = fetch.run("https://example.com/long")

    assert len(out["text"]) == fetch.MAX_CONTENT_CHARS
    assert out["char_count"] == fetch.MAX_CONTENT_CHARS


def test_fetch_tool_schema_is_anthropic_shape():
    schema = fetch.TOOL_SCHEMA
    assert schema["name"] == "fetch"
    assert "url" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["url"]
