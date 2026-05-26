"""Agent tool-call loop. Direct Anthropic SDK — no framework (D2).

~150 lines, reads top-to-bottom. Extension = add an entry to TOOL_REGISTRY and
mention the tool in agent/prompts.py SYSTEM_PROMPT. Iteration cap is a kill
switch (D7); on cap hit the run is marked partial and a fallback summarize is
attempted with whatever was fetched.
"""

import json
import time
from types import ModuleType

from agent import llm, prompts, storage
from agent.tools import fetch, search, summarize

MAX_ITERATIONS = 8  # 3 phases × 2-3 calls ceiling. Kill switch, not design target.
MAX_TOKENS = 2048


# Registry stores (schema, module) — NOT (schema, function). Module lookup happens
# at call time so tests can patch `agent.tools.<name>.run` and the loop picks up
# the patched version. New tool = new entry here + describe it in prompts.SYSTEM_PROMPT.
TOOL_REGISTRY: dict[str, tuple[dict, ModuleType]] = {
    "search": (search.TOOL_SCHEMA, search),
    "fetch": (fetch.TOOL_SCHEMA, fetch),
    "summarize": (summarize.TOOL_SCHEMA, summarize),
}


def _dispatch_tool(name: str, tool_input: dict) -> dict:
    """Execute one tool call. Loud failure → error dict, never raise to loop."""
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    _, module = TOOL_REGISTRY[name]
    try:
        return module.run(**tool_input)
    except TypeError as exc:
        return {"error": f"bad tool input for {name}: {exc}"}
    except Exception as exc:  # surface in trace, do not crash the loop
        return {"error": f"{name} raised: {exc!r}"}


def _tool_result_block(tool_use_id: str, content: dict) -> dict:
    """Serialize tool output for the next assistant turn."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(content, ensure_ascii=False)[:60_000],
    }


def run(user_prompt: str) -> dict:
    """Run the agent end-to-end on one user prompt. Returns briefing + trace."""
    started = time.time()
    tools = [schema for schema, _ in TOOL_REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    tool_calls: list[dict] = []
    briefing_text = ""
    status = "incomplete"
    iteration = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = llm.call_with_retry({
            "model": llm.SONNET_MODEL,
            "max_tokens": MAX_TOKENS,
            "tools": tools,
            "system": prompts.SYSTEM_PROMPT,
            "messages": messages,
        })

        # The assistant's full content block must go back into history for the
        # next tool_result turn to correlate with tool_use_ids.
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text"]

        if not tool_uses:
            # Model produced a final text. Prefer the most recent summarize result;
            # fall back to the model's text if it summarized inline.
            for call in reversed(tool_calls):
                if call["name"] == "summarize" and call.get("result", {}).get("briefing"):
                    briefing_text = call["result"]["briefing"]
                    break
            if not briefing_text:
                briefing_text = "\n".join(text_blocks).strip()
            status = "complete"
            break

        # Execute every tool the model requested this turn.
        tool_result_blocks = []
        for tool_use in tool_uses:
            result = _dispatch_tool(tool_use.name, dict(tool_use.input))
            tool_calls.append({
                "iteration": iteration,
                "name": tool_use.name,
                "input": dict(tool_use.input),
                # Keep summarize result for briefing extraction; others stay light.
                "result": result if tool_use.name == "summarize" else None,
                "error": result.get("error") if isinstance(result, dict) else None,
            })
            tool_result_blocks.append(_tool_result_block(tool_use.id, result))

        messages.append({"role": "user", "content": tool_result_blocks})

    elapsed = time.time() - started

    if status != "complete":
        status = "partial_max_iterations"
        if not briefing_text:
            # Last-resort: feed whatever we fetched into summarize directly.
            fetched_docs = []
            for call in tool_calls:
                if call["name"] == "fetch":
                    cached = storage.load_fetched(call["input"].get("url", ""))
                    if cached and cached.get("text"):
                        fetched_docs.append(cached)
            fallback = summarize.run(prompt=user_prompt, documents=fetched_docs)
            briefing_text = fallback.get("briefing", "")

    briefing_path = storage.save_briefing(user_prompt, briefing_text)
    storage.log_run({
        "prompt": user_prompt,
        "status": status,
        "iterations": iteration,
        "elapsed_seconds": round(elapsed, 2),
        "tool_calls": [
            {"i": c["iteration"], "name": c["name"], "error": c["error"]}
            for c in tool_calls
        ],
        "briefing_path": str(briefing_path.relative_to(storage.ROOT)) if briefing_path else None,
    })

    return {
        "briefing": briefing_text,
        "briefing_path": briefing_path,
        "status": status,
        "iterations": iteration,
        "elapsed_seconds": round(elapsed, 2),
        "tool_call_count": len(tool_calls),
    }
