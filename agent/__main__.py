"""CLI entry: python -m agent "<prompt>"

Single positional arg per PDF spec (R5). Briefing prints to stdout; status to
stderr. Exit 0 on complete, 1 on partial (iteration cap hit or rate-limited).
"""

import sys

from agent import loop


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or not args[0].strip():
        print('usage: python -m agent "<prompt>"', file=sys.stderr)
        return 2

    prompt = " ".join(args).strip()
    result = loop.run(prompt)

    print(result["briefing"])
    print(
        f"\n---\nSaved to {result['briefing_path']} · "
        f"status={result['status']} · iterations={result['iterations']} · "
        f"{result['elapsed_seconds']}s · {result['tool_call_count']} tool calls",
        file=sys.stderr,
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
