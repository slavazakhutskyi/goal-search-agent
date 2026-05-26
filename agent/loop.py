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

MAX_ITERATIONS = 12  # Originally 8 from first-principles; observed prompt #1 (RapidSOS 7-day)
                      # hit the cap with a still-valid briefing via the partial fallback. Raised
                      # to 12 after the U5 iteration-cap measurement step (see plan U4 verification).
                      # Kill switch, not design target.
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
    last_summarize_briefing: str | None = None  # set after every successful summarize dispatch
    briefing_text = ""
    status = "incomplete"
    llm_error: str | None = None
    iteration = 0
    total_input_tokens = 0
    total_output_tokens = 0

    storage.log_event("INFO", "agent run start", prompt=repr(user_prompt)[:80], model=llm.SONNET_MODEL)

    for iteration in range(1, MAX_ITERATIONS + 1):
        try:
            response = llm.call_with_retry({
                "model": llm.SONNET_MODEL,
                "max_tokens": MAX_TOKENS,
                "tools": tools,
                "system": prompts.SYSTEM_PROMPT,
                "messages": messages,
            })
        except Exception as exc:
            # Non-retryable LLM error (auth, bad request, schema, network exhausted).
            # Loud-log and break so we still write runs.jsonl + a partial briefing.
            llm_error = f"{type(exc).__name__}: {exc}"
            status = "partial_llm_error"
            storage.log_event("ERROR", "LLM call failed", iteration=iteration, error=llm_error)
            break

        usage = getattr(response, "usage", None)
        if usage is not None:
            total_input_tokens += getattr(usage, "input_tokens", 0)
            total_output_tokens += getattr(usage, "output_tokens", 0)

        # Assistant content goes back into history for tool_result correlation.
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text"]

        if not tool_uses:
            # Final text turn. Prefer the most recent successful summarize result;
            # fall back to the model's inline text if no summarize ever ran.
            if last_summarize_briefing:
                briefing_text = last_summarize_briefing
            elif text_blocks:
                briefing_text = "\n".join(text_blocks).strip()
            # If both empty, we leave briefing_text="" and fall into the partial branch
            # rather than persisting an empty briefing as "complete".
            if briefing_text:
                status = "complete"
                break
            status = "partial_empty_response"
            break

        # Execute every tool the model requested this turn.
        tool_result_blocks = []
        for tool_use in tool_uses:
            tool_started = time.time()
            result = _dispatch_tool(tool_use.name, dict(tool_use.input))
            tool_elapsed = round(time.time() - tool_started, 2)
            if tool_use.name == "summarize" and isinstance(result, dict) and result.get("briefing"):
                last_summarize_briefing = result["briefing"]
            err = result.get("error") if isinstance(result, dict) else None
            tool_calls.append({
                "iteration": iteration,
                "name": tool_use.name,
                "input": dict(tool_use.input),
                "elapsed_seconds": tool_elapsed,
                "error": err,
            })
            level = "WARN" if err else "INFO"
            storage.log_event(level, f"tool {tool_use.name}", iteration=iteration, elapsed=tool_elapsed, error=err or "")
            tool_result_blocks.append(_tool_result_block(tool_use.id, result))

        messages.append({"role": "user", "content": tool_result_blocks})

    elapsed = time.time() - started

    if status == "incomplete":
        status = "partial_max_iterations"

    if not briefing_text:
        # Reuse a successful summarize if the model called it before hitting cap/error.
        if last_summarize_briefing:
            briefing_text = last_summarize_briefing
        else:
            # Last-resort: feed whatever we fetched into summarize directly.
            fetched_docs = []
            for call in tool_calls:
                if call["name"] == "fetch":
                    cached = storage.load_fetched(call["input"].get("url", ""))
                    if cached and cached.get("text"):
                        fetched_docs.append(cached)
            try:
                fallback = summarize.run(prompt=user_prompt, documents=fetched_docs)
                briefing_text = fallback.get("briefing", "")
            except Exception as exc:
                briefing_text = (
                    f"# Briefing — {user_prompt}\n\n"
                    f"## TL;DR\n\nRun terminated with status `{status}`. "
                    f"Fallback summarize also failed: {type(exc).__name__}: {exc}\n"
                )

    briefing_path = storage.save_briefing(user_prompt, briefing_text)
    log_entry = {
        "prompt": user_prompt,
        "status": status,
        "iterations": iteration,
        "elapsed_seconds": round(elapsed, 2),
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
        },
        "tool_calls": [
            {
                "iteration": c["iteration"],
                "name": c["name"],
                "elapsed_seconds": c.get("elapsed_seconds", 0),
                "error": c["error"],
            }
            for c in tool_calls
        ],
        "briefing_path": str(briefing_path.relative_to(storage.ROOT)) if briefing_path else None,
    }
    if llm_error:
        log_entry["llm_error"] = llm_error
    storage.log_run(log_entry)
    storage.log_event(
        "INFO", f"agent run end status={status}",
        iterations=iteration, elapsed=round(elapsed, 2),
        tokens_in=total_input_tokens, tokens_out=total_output_tokens,
        tools=len(tool_calls),
    )

    return {
        "briefing": briefing_text,
        "briefing_path": briefing_path,
        "status": status,
        "iterations": iteration,
        "elapsed_seconds": round(elapsed, 2),
        "tool_call_count": len(tool_calls),
        "tokens": {"input": total_input_tokens, "output": total_output_tokens},
    }
