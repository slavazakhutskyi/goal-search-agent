"""Pytest configuration and shared fixtures.

`temp_data_dir` redirects agent.storage's flat-file paths to a tmp_path so tests
never touch the real data/ directory. Fixtures for U4 (mock_anthropic_briefing,
mock_search_results, mock_fetch_html) will be added when those tests need them.
"""

import pytest


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
