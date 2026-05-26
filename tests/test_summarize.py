"""Tests for agent.tools.summarize."""

from agent.tools import summarize
from tests.conftest import make_response, make_text_block


def test_summarize_with_documents(mock_llm_queue):
    """Mocked Anthropic returns markdown briefing → summarize.run returns it."""
    canned_briefing = (
        "# Briefing — Test\n\n"
        "## TL;DR\nSomething happened.\n\n"
        "## Key themes\n- A theme [^1]\n\n"
        "## Sentiment\n**Neutral** — calm coverage.\n\n"
        "## Sources\n[^1]: [Title](https://example.com) — example.com, 2026-05-20\n"
    )
    mock_llm_queue.append(make_response([make_text_block(canned_briefing)]))

    out = summarize.run(
        prompt="Test prompt",
        documents=[{
            "url": "https://example.com/x",
            "text": "Article body.",
            "metadata": {"source_domain": "example.com", "title": "X", "publish_date": "2026-05-20"},
        }],
    )

    assert "TL;DR" in out["briefing"]
    assert "Sources" in out["briefing"]


def test_summarize_empty_documents(mock_llm_queue):
    """Empty docs → stub briefing returned, NO Anthropic call."""
    out = summarize.run(prompt="Find unicorns", documents=[])

    assert "0 sources" in out["briefing"]
    assert "No sources retrieved" in out["briefing"]
    # mock_llm_queue should still be empty (queue never popped)
    assert mock_llm_queue == []


def test_summarize_formats_documents_with_labels(mocker):
    """Two docs → user message sent to LLM contains [Document 1] and [Document 2] labels."""
    spy = mocker.patch(
        "agent.llm.call_with_retry",
        return_value=make_response([make_text_block("# Briefing — Test\n\n## TL;DR\nok")]),
    )

    summarize.run(
        prompt="X",
        documents=[
            {"url": "https://a.com", "text": "a body", "metadata": {"title": "A", "publish_date": "2026-05-19"}},
            {"url": "https://b.com", "text": "b body", "metadata": {"title": "B", "publish_date": None}},
        ],
    )

    create_kwargs = spy.call_args.args[0]
    user_msg = create_kwargs["messages"][0]["content"]
    assert "[Document 1]" in user_msg
    assert "[Document 2]" in user_msg
    assert "https://a.com" in user_msg
    assert "https://b.com" in user_msg
    assert "unknown" in user_msg  # B has no publish_date


def test_summarize_tool_schema_is_anthropic_shape():
    schema = summarize.TOOL_SCHEMA
    assert schema["name"] == "summarize"
    assert set(schema["input_schema"]["required"]) == {"prompt", "documents"}
