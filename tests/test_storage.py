"""Tests for agent.storage. Pure functions — TDD per U3 Execution note."""

import json

from agent import storage


def test_url_hash_deterministic():
    """Same URL → same hash; different URLs → different hashes."""
    h1 = storage.url_hash("https://example.com/a")
    h2 = storage.url_hash("https://example.com/a")
    h3 = storage.url_hash("https://example.com/b")
    assert h1 == h2
    assert h1 != h3


def test_url_hash_length():
    """Hash is exactly 16 hex chars (SHA-256 truncated)."""
    h = storage.url_hash("https://example.com")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_save_load_fetched_roundtrip(temp_data_dir):
    """save_fetched → load_fetched returns equivalent dict."""
    url = "https://example.com/article"
    payload = {"url": url, "text": "hello world", "metadata": {"source_domain": "example.com"}}

    storage.save_fetched(url, payload)
    loaded = storage.load_fetched(url)

    assert loaded == payload


def test_load_fetched_missing_returns_none(temp_data_dir):
    """Unsaved URL → load returns None (not an exception)."""
    assert storage.load_fetched("https://never-saved.example.com") is None


def test_slugify_normalization():
    """Punctuation/case normalized to lowercase kebab; length capped."""
    assert storage.slugify("Give me a briefing!") == "give-me-a-briefing"
    assert storage.slugify("Multi   spaces   here") == "multi-spaces-here"
    long = storage.slugify("x" * 200)
    assert len(long) <= 50
    assert storage.slugify("") == "briefing"  # never empty


def test_save_briefing_creates_file_with_timestamp_slug(temp_data_dir):
    """save_briefing returns Path; filename includes timestamp + slug; content written."""
    path = storage.save_briefing("Give me a briefing on RapidSOS", "# Briefing\n\nHello.")

    assert path.exists()
    assert path.parent == temp_data_dir / "briefings"
    assert "give-me-a-briefing-on-rapidsos" in path.name
    assert path.suffix == ".md"
    assert path.read_text() == "# Briefing\n\nHello."


def test_log_run_appends_jsonl(temp_data_dir):
    """Two log_run calls → two lines, each valid JSON; timestamp added automatically."""
    storage.log_run({"prompt": "a", "status": "complete", "iterations": 3})
    storage.log_run({"prompt": "b", "status": "partial", "iterations": 8})

    runs_path = temp_data_dir / "logs" / "runs.jsonl"
    assert runs_path.exists()
    lines = runs_path.read_text().strip().split("\n")
    assert len(lines) == 2

    entry_a = json.loads(lines[0])
    entry_b = json.loads(lines[1])
    assert entry_a["prompt"] == "a"
    assert entry_b["prompt"] == "b"
    assert "timestamp" in entry_a and "timestamp" in entry_b
