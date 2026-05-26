"""Pytest configuration and shared fixtures.

- temp_data_dir: redirect storage flat-file paths to tmp_path (U3+)
- make_text_block / make_tool_use_block / make_response: mock anthropic SDK shape
- mock_llm_queue: queue-based mock for agent.llm.call_with_retry
"""

import types

import pytest


# ---------- Storage isolation (U3) ----------

@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Redirect storage.DATA/FETCHED/BRIEFINGS/LOGS to tmp_path for the test."""
    from agent import storage

    fetched = tmp_path / "fetched"
    briefings = tmp_path / "briefings"
    logs = tmp_path / "logs"
    for p in (fetched, briefings, logs):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage, "DATA", tmp_path)
    monkeypatch.setattr(storage, "FETCHED", fetched)
    monkeypatch.setattr(storage, "BRIEFINGS", briefings)
    monkeypatch.setattr(storage, "LOGS", logs)
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    return tmp_path


# ---------- Anthropic SDK shape mocks (U4) ----------

def make_text_block(text: str):
    """Mock an anthropic text block."""
    return types.SimpleNamespace(type="text", text=text)


def make_tool_use_block(tool_id: str, name: str, tool_input: dict):
    """Mock an anthropic tool_use block."""
    return types.SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def make_response(content_blocks: list):
    """Mock an anthropic Message with the given content blocks."""
    return types.SimpleNamespace(content=content_blocks, stop_reason="end_turn")


@pytest.fixture
def mock_llm_queue(mocker):
    """Queue-based mock for agent.llm.call_with_retry.

    Tests append response objects to the returned list; each call_with_retry
    invocation pops the next response. Raises if queue exhausted.
    """
    queue: list = []

    def fake_call(create_kwargs, max_retries=3):
        if not queue:
            raise RuntimeError(
                "mock_llm_queue exhausted — test did not queue enough responses"
            )
        return queue.pop(0)

    mocker.patch("agent.llm.call_with_retry", side_effect=fake_call)
    return queue
