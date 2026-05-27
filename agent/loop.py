"""Agent tool-call loop. Direct Anthropic SDK — no framework (D2).

~150 lines, reads top-to-bottom. Extension = add an entry to TOOL_REGISTRY and
mention the tool in agent/prompts.py SYSTEM_PROMPT. Iteration cap is a kill
switch (D7); on cap hit the run is marked partial and a fallback summarize is
attempted with whatever was fetched.
"""

import json
import time
from types import ModuleType

from agent import evaluate, llm, prompts, storage
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


def _collect_fetched_docs(tool_calls: list[dict]) -> list[dict]:
    """Read cached docs for every successful fetch in the current run.

    Auto-attach helper for the summarize-without-documents path (U0 from
    docs/plans/2026-05-27-002...). Mirrors the existing fallback pattern at
    the bottom of `run()` so both paths use the same definition of
    "successfully fetched" — has text, no error.
    """
    docs: list[dict] = []
    for call in tool_calls:
        if call.get("name") != "fetch":
            continue
        url = (call.get("input") or {}).get("url", "")
        if not url:
            continue
        cached = storage.load_fetched(url)
        if cached and cached.get("text") and not cached.get("error"):
            docs.append(cached)
    return docs


def _tool_result_block(tool_use_id: str, content: dict) -> dict:
    """Serialize tool output for the next assistant turn."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(content, ensure_ascii=False)[:60_000],
    }


def run(user_prompt: str, *, self_eval: bool = True) -> dict:
    """Run the agent end-to-end on one user prompt. Returns briefing + trace.

    `self_eval=True` (default) runs a self-critique pass after the briefing is
    saved and appends one line to data/logs/evals.jsonl. The next run reads
    those lessons and appends them to SYSTEM_PROMPT — that is the compounding
    feedback loop. Tests pass `self_eval=False` to keep runs deterministic.
    """
    started = time.time()
    tools = [schema for schema, _ in TOOL_REGISTRY.values()]
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    # Compounding loop: append few-shot demonstrations to SYSTEM_PROMPT.
    # demonstrations_block returns "" until ≥3 composite-v1 edit_history
    # records exist (metric_version gate per U3 — prevents teaching the old
    # saturated-metric biases via demonstrations from pre-composite runs).
    # Until then SYSTEM_PROMPT is unchanged from the static base.
    system_prompt = prompts.SYSTEM_PROMPT
    if self_eval:
        demo_block = evaluate.demonstrations_block()
        system_prompt = system_prompt + demo_block

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
                "system": system_prompt,
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
            tool_input = dict(tool_use.input)
            # U0 auto-attach: if the model called summarize without documents,
            # inject all successfully-fetched docs from this run before dispatch.
            # Eliminates the summarize_missing_documents failure pattern that
            # fired in 6 of 7 historical runs.
            if tool_use.name == "summarize" and "documents" not in tool_input:
                tool_input["documents"] = _collect_fetched_docs(tool_calls)
            result = _dispatch_tool(tool_use.name, tool_input)
            tool_elapsed = round(time.time() - tool_started, 2)
            if tool_use.name == "summarize" and isinstance(result, dict) and result.get("briefing"):
                last_summarize_briefing = result["briefing"]
            err = result.get("error") if isinstance(result, dict) else None
            # Record the as-dispatched input (post auto-attach). For summarize
            # auto-attach we collapse the docs to a count so the log stays
            # compact and the URL list isn't duplicated from fetch records.
            recorded_input = dict(tool_use.input)
            if tool_use.name == "summarize" and "documents" not in tool_use.input and "documents" in tool_input:
                recorded_input["_auto_attached_docs"] = len(tool_input["documents"])
            tool_calls.append({
                "iteration": iteration,
                "name": tool_use.name,
                "input": recorded_input,
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
        elif llm_error:
            # LLM is already broken — skip the fallback summarize call (it would
            # just re-fail through the same retry budget and waste seconds).
            briefing_text = (
                f"# Briefing — {user_prompt}\n\n"
                f"## TL;DR\n\nRun terminated with status `{status}`. "
                f"LLM error: {llm_error}\n"
            )
        else:
            # Last-resort: feed whatever we fetched into summarize directly.
            # Filter to docs with real content — empty/error payloads pollute output.
            fetched_docs = []
            for call in tool_calls:
                if call["name"] == "fetch":
                    cached = storage.load_fetched(call["input"].get("url", ""))
                    if cached and cached.get("text") and not cached.get("error"):
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

    # Self-evaluation pass. Wrapped — eval failure must never break the run.
    eval_record: dict | None = None
    if self_eval:
        try:
            eval_record = evaluate.run_self_eval(log_entry, briefing_text)
            evaluate.save_eval(eval_record)
            storage.log_event(
                "INFO", "self-eval done",
                score=eval_record["structure"]["score"],
                issues=len(eval_record["trace_issues"]),
            )
        except Exception as exc:
            storage.log_event("WARN", "self-eval failed", error=f"{type(exc).__name__}: {exc}")

    return {
        "briefing": briefing_text,
        "briefing_path": briefing_path,
        "status": status,
        "iterations": iteration,
        "elapsed_seconds": round(elapsed, 2),
        "tool_call_count": len(tool_calls),
        "tokens": {"input": total_input_tokens, "output": total_output_tokens},
        "eval": eval_record,
    }
