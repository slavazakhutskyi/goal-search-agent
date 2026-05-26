"""CLI: python -m eval [--no-judge] [path-to-briefings-dir]

Runs the hybrid eval (deterministic schema + LLM judge) on every .md in the
given dir. Prints a per-file scorecard + aggregate. With --no-judge, runs
deterministic only — no API calls, free.
"""

import json
import sys
from pathlib import Path

from eval import checks


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    use_judge = "--no-judge" not in args
    args = [a for a in args if a != "--no-judge"]
    target = Path(args[0]) if args else Path("data/briefings")

    if not target.exists():
        print(f"no such path: {target}", file=sys.stderr)
        return 2

    files = sorted(target.glob("*.md")) if target.is_dir() else [target]
    if not files:
        print(f"no .md files in {target}", file=sys.stderr)
        return 1

    results = []
    for f in files:
        result = checks.evaluate_briefing_file(f, use_llm_judge=use_judge)
        results.append(result)
        det = result["deterministic"]
        print(f"\n=== {f.name} ===")
        print(f"  deterministic: {det['score']}/{det['max_score']}  passed={det['passed']}  sentiment={det['sentiment_class']}")
        for check, ok in det["checks"].items():
            mark = "✓" if ok else "✗"
            print(f"    {mark} {check}")
        if use_judge and "llm_judge" in result:
            j = result["llm_judge"]
            if "error" in j:
                print(f"  llm_judge: ERROR — {j['error']}")
            else:
                print(f"  llm_judge: coherence={j.get('coherence','?')}  "
                      f"citations={j.get('citation_discipline','?')}  "
                      f"sentiment_fit={j.get('sentiment_fit','?')}")
                if "note" in j:
                    print(f"    note: {j['note']}")

    # Aggregate
    det_pass = sum(1 for r in results if r["deterministic"]["passed"])
    print(f"\n--- aggregate ---")
    print(f"  deterministic pass rate: {det_pass}/{len(results)}")
    if use_judge:
        coh = [r.get("llm_judge", {}).get("coherence") for r in results]
        coh = [c for c in coh if isinstance(c, int)]
        if coh:
            print(f"  judge coherence mean: {sum(coh)/len(coh):.2f}")

    # Machine-readable JSON for downstream tooling
    out_path = Path("data/logs/eval-latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  full results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
