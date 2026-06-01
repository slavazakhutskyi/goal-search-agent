"""CLI entry: python -m agent "<prompt>"

Single positional arg per PDF spec (R5). Briefing prints to stdout; status to
stderr. Exit 0 on complete, 1 on partial (iteration cap hit or rate-limited).
"""

import sys

from agent import loop


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]

    self_eval = True
    if "--no-eval" in args:
        self_eval = False
        args = [a for a in args if a != "--no-eval"]

    if not args or not args[0].strip():
        print('usage: python -m agent [--no-eval] "<goal>"', file=sys.stderr)
        return 2

    prompt = " ".join(args).strip()
    result = loop.run(prompt, self_eval=self_eval)

    print(result["briefing"])
    print(
        f"\n---\nSaved to {result['briefing_path']} · "
        f"status={result['status']} · iterations={result['iterations']} · "
        f"{result['elapsed_seconds']}s · {result['tool_call_count']} tool calls",
        file=sys.stderr,
    )
    if result.get("eval"):
        ev = result["eval"]
        s = ev["structure"]
        n_issues = len(ev["trace_issues"])
        judge = ev.get("judge") or {}
        judge_score = judge.get("usefulness") if isinstance(judge, dict) else None
        print(
            f"Self-eval: structure {s['score']}/5 · "
            f"{n_issues} trace issue(s) · "
            f"judge usefulness: {judge_score if judge_score is not None else 'n/a'}",
            file=sys.stderr,
        )
        for issue in ev["trace_issues"]:
            print(f"  [{issue['tag']}] {issue['suggestion']}", file=sys.stderr)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
