"""Tests for agent.__main__ CLI."""

from pathlib import Path

from agent import __main__ as cli


def test_cli_with_prompt_arg(capsys, mocker):
    """Valid prompt → loop.run called, briefing printed to stdout, exit 0."""
    mocker.patch("agent.loop.run", return_value={
        "briefing": "# Briefing — Test\n\nbody",
        "briefing_path": Path("/tmp/briefing.md"),
        "status": "complete",
        "iterations": 3,
        "elapsed_seconds": 1.2,
        "tool_call_count": 4,
    })

    exit_code = cli.main(["Test prompt"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# Briefing — Test" in captured.out
    assert "status=complete" in captured.err


def test_cli_no_args_returns_usage_error(capsys):
    """No args → usage on stderr, exit 2."""
    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "usage:" in captured.err


def test_cli_partial_status_exit_1(mocker):
    """status=partial_max_iterations → exit 1 (signals partial result)."""
    mocker.patch("agent.loop.run", return_value={
        "briefing": "partial briefing",
        "briefing_path": Path("/tmp/briefing.md"),
        "status": "partial_max_iterations",
        "iterations": 8,
        "elapsed_seconds": 5.0,
        "tool_call_count": 8,
    })

    exit_code = cli.main(["Test prompt"])

    assert exit_code == 1


def test_cli_joins_multiple_args(capsys, mocker):
    """Multiple positional args → joined as single prompt."""
    spy = mocker.patch("agent.loop.run", return_value={
        "briefing": "ok",
        "briefing_path": Path("/tmp/x.md"),
        "status": "complete",
        "iterations": 1,
        "elapsed_seconds": 0.1,
        "tool_call_count": 0,
    })

    cli.main(["Give", "me", "a", "briefing"])

    spy.assert_called_once_with("Give me a briefing")
