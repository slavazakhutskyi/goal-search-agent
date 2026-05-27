"""Anthropic SDK wrapper. Single Sonnet model (D3); model ID resolved here.

`call_with_retry` provides exponential backoff on rate limits and transient
errors. Loud failure — re-raises after retries so the agent loop can log
the run as partial rather than silently swallowing.
"""

import os
import time

import anthropic

# Hardcoded per spec (D3 — single model, no provider abstraction).
# Swapped to Haiku 2026-05-27 to fit remaining $1.95 budget for live demo.
# Constant kept as SONNET_MODEL for blast-radius minimization; rename later.
SONNET_MODEL = "claude-haiku-4-5-20251001"

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    """Lazy singleton. Raises if ANTHROPIC_API_KEY is missing."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. See .env.example.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_with_retry(create_kwargs: dict, max_retries: int = 3):
    """Call client.messages.create with exponential backoff. Re-raises on final failure."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client().messages.create(**create_kwargs)
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= 2
    assert last_exc is not None
    raise last_exc
