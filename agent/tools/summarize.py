"""Summarize tool. Single Sonnet call with all fetched documents as context.

Spec-mandated (R4). Returns the final markdown briefing. Called exactly once per
run (the system prompt enforces this).
"""

from datetime import datetime, timezone

from agent import llm, prompts


TOOL_SCHEMA = {
    "name": "summarize",
    "description": (
        "Produce the final markdown briefing from the list of fetched documents. "
        "Call this exactly once, after gathering 4-8 sources via search and fetch. "
        "Returns the briefing as a markdown string — return it verbatim to the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The original user prompt verbatim.",
            },
            "documents": {
                "type": "array",
                "description": "List of fetched documents (with text + metadata).",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                },
            },
        },
        "required": ["prompt", "documents"],
    },
}


def _format_documents(documents: list[dict]) -> str:
    """Render docs as labeled blocks the summarizer can cite by number."""
    chunks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.get("metadata") or {}
        chunks.append(
            f"[Document {i}]\n"
            f"URL: {doc.get('url', '')}\n"
            f"Title: {meta.get('title') or '(no title)'}\n"
            f"Domain: {meta.get('source_domain') or '(unknown)'}\n"
            f"Publish date: {meta.get('publish_date') or 'unknown'}\n"
            f"Text:\n{(doc.get('text') or '').strip()}\n"
            f"---"
        )
    return "\n\n".join(chunks)


def run(prompt: str, documents: list[dict]) -> dict:
    """Returns {'briefing': <markdown string>}. Empty docs → stub no-sources briefing."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not documents:
        return {
            "briefing": (
                f"# Briefing — {prompt}\n"
                f"*Generated {ts} · 0 sources · sentiment: neutral*\n\n"
                f"## TL;DR\n\nNo sources retrieved. Search returned no usable results for this prompt.\n"
            ),
        }

    prompt_short = prompt[:80] + ("..." if len(prompt) > 80 else "")
    rendered_docs = _format_documents(documents)

    user_message = prompts.SUMMARIZE_PROMPT.format(
        prompt=prompt,
        prompt_short=prompt_short,
        timestamp=ts,
        n_sources=len(documents),
        documents=rendered_docs,
    )

    response = llm.call_with_retry({
        "model": llm.SONNET_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user_message}],
    })

    briefing_text = ""
    for block in response.content:
        if block.type == "text":
            briefing_text += block.text

    return {"briefing": briefing_text.strip()}
