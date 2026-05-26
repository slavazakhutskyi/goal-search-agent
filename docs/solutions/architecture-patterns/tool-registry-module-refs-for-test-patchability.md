---
module: agent_loop
date: 2026-05-26
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - "Anthropic SDK tool-call loop with multiple tools"
  - "Test suite needs to mock individual tool implementations"
  - "Plan to live-extend the agent by adding more tools"
tags:
  - anthropic-sdk
  - tool-calling
  - testing
  - registry-pattern
  - pytest-monkeypatch
related_components:
  - testing_framework
  - tooling
---

# Tool registry stores module references, not function references — for test patchability + live extensibility

## Context

A direct-Anthropic-SDK tool-call agent dispatches user-named tools (search, fetch, summarize, etc.) via a registry. The naive shape is `dict[name, callable]`:

```python
TOOL_REGISTRY = {
    "search": (search.TOOL_SCHEMA, search.run),
    "fetch": (fetch.TOOL_SCHEMA, fetch.run),
    "summarize": (summarize.TOOL_SCHEMA, summarize.run),
}
```

This captures a **direct function reference** at import time. The dispatch function calls `runner(**input)` directly.

The problem surfaces in tests: `mocker.patch("agent.tools.summarize.run", return_value=...)` replaces the `run` attribute on the module — but the registry's cached function reference is unchanged. The mocked tool is never called, the real implementation runs, the test fails or (worse) silently bypasses the mock.

## Guidance

Store **module references** in the registry, not function references. Resolve the function at call time:

```python
from types import ModuleType

TOOL_REGISTRY: dict[str, tuple[dict, ModuleType]] = {
    "search": (search.TOOL_SCHEMA, search),
    "fetch": (fetch.TOOL_SCHEMA, fetch),
    "summarize": (summarize.TOOL_SCHEMA, summarize),
}

def _dispatch_tool(name: str, tool_input: dict) -> dict:
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    _, module = TOOL_REGISTRY[name]
    return module.run(**tool_input)
```

Now `mocker.patch("agent.tools.summarize.run", return_value=...)` updates `summarize.run`, and `module.run` looks up the patched attribute fresh on every dispatch.

## Why This Matters

Two load-bearing properties:

1. **Test patchability.** Without module-ref dispatch, every test that wants to mock a tool's implementation needs to either patch `TOOL_REGISTRY` directly (brittle) or restructure the loop (over-engineering). Module-ref dispatch makes the standard `mocker.patch` pattern work without ceremony.

2. **Hot-swappability for live extension.** During a live coding interview / dev session, adding a new tool or temporarily replacing one (e.g., A/B testing two summarize prompts) is one `module.run` swap, not a registry rebuild.

The cost is trivial — 5 lines of code, one extra indirection at dispatch time. The benefit recurs every time tests run or a tool needs replacement.

## When to Apply

- Direct-SDK agent with a `TOOL_REGISTRY` mapping → always use module refs
- LangGraph / LangChain agents don't need this — they have their own tool-binding lifecycle
- For single-tool agents the pattern is overkill — just import and call directly

## Examples

**Before (broken under pytest-mock):**

```python
# loop.py
from agent.tools import search
TOOL_REGISTRY = {"search": (search.TOOL_SCHEMA, search.run)}

def _dispatch_tool(name, input):
    _, runner = TOOL_REGISTRY[name]
    return runner(**input)
```

```python
# test_loop.py — this test SILENTLY uses the real search.run
def test_loop_with_mocked_search(mocker):
    mocker.patch("agent.tools.search.run", return_value={"results": [...]})
    loop.run("test prompt")  # calls REAL search.run, mock never triggered
```

**After (works correctly):**

```python
# loop.py
from agent.tools import search
from types import ModuleType
TOOL_REGISTRY: dict[str, tuple[dict, ModuleType]] = {
    "search": (search.TOOL_SCHEMA, search),
}

def _dispatch_tool(name, input):
    _, module = TOOL_REGISTRY[name]
    return module.run(**input)  # resolves at call time
```

```python
# test_loop.py — mock now intercepts dispatch
def test_loop_with_mocked_search(mocker):
    mocker.patch("agent.tools.search.run", return_value={"results": [...]})
    loop.run("test prompt")  # calls MOCKED search.run
```

## Prevention

When writing the first test that mocks a tool, verify the mock is actually intercepted by adding `mocker.patch(..., side_effect=AssertionError("mock not called"))` and confirming the test FAILS with that error. If it passes anyway, the registry is caching function refs.
