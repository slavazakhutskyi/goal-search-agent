"""Tests for agent.tools.search.

Test-first per U2 Execution note. Mocks httpx GET — does not require a running
SearXNG instance. Covers happy path + 3 graceful-failure paths (R13).
"""

import httpx
import pytest

from agent.tools import search


def _mock_search_response(mocker, *, json_data=None, raise_exc=None, text=""):
    """Build a httpx.get mock that returns a Response-like or raises."""
    if raise_exc is not None:
        mocker.patch("agent.tools.search.httpx.get", side_effect=raise_exc)
        return

    response = mocker.MagicMock(spec=httpx.Response)
    response.raise_for_status = mocker.MagicMock(return_value=None)
    if json_data is None:
        response.json = mocker.MagicMock(side_effect=ValueError("not json"))
    else:
        response.json = mocker.MagicMock(return_value=json_data)
    response.text = text
    mocker.patch("agent.tools.search.httpx.get", return_value=response)


def test_search_happy_path(mocker):
    """SearXNG returns 3 results → run normalizes to title/url/snippet list."""
    _mock_search_response(
        mocker,
        json_data={
            "results": [
                {"title": "RapidSOS launches X", "url": "https://example.com/1", "content": "snippet 1"},
                {"title": "Public safety AI", "url": "https://example.com/2", "content": "snippet 2"},
                {"title": "911 modernization", "url": "https://example.com/3", "content": "snippet 3"},
            ]
        },
    )

    out = search.run("rapidsos")

    assert "error" not in out
    assert out["query"] == "rapidsos"
    assert len(out["results"]) == 3
    for item in out["results"]:
        assert set(item.keys()) >= {"title", "url", "snippet"}
    assert out["results"][0]["url"] == "https://example.com/1"


def test_search_empty_results(mocker):
    """Empty result set is not an error — return empty list with query echoed."""
    _mock_search_response(mocker, json_data={"results": []})

    out = search.run("obscure query that finds nothing")

    assert "error" not in out
    assert out["results"] == []
    assert out["query"] == "obscure query that finds nothing"


def test_search_http_error(mocker):
    """httpx errors degrade gracefully — return error dict, never raise to caller."""
    _mock_search_response(mocker, raise_exc=httpx.TimeoutException("read timeout"))

    out = search.run("rapidsos")

    assert "error" in out
    assert out["results"] == []
    assert "timeout" in out["error"].lower() or "TimeoutException" in out["error"]


def test_search_non_json_response(mocker):
    """If SearXNG returns text/html instead of JSON, surface as error not crash."""
    _mock_search_response(mocker, json_data=None, text="<html>SearXNG error page</html>")

    out = search.run("rapidsos")

    assert "error" in out
    assert out["results"] == []


def test_search_tool_schema_is_anthropic_shape():
    """TOOL_SCHEMA must follow Anthropic tool-use format the loop expects."""
    schema = search.TOOL_SCHEMA
    assert schema["name"] == "search"
    assert isinstance(schema["description"], str) and len(schema["description"]) > 20
    assert schema["input_schema"]["type"] == "object"
    assert "query" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["query"]
