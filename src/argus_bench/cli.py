import argparse
import json
import sys

from .check import check


def main(argv=None):
    ap = argparse.ArgumentParser(prog="argus-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check that a rollout file is well-formed")
    c.add_argument("files", nargs="+")
    sc = sub.add_parser("check-submission", help="check a submission folder or .zip (rollouts/, labels.json, verifier.py, ...)")
    sc.add_argument("path")
    sc.add_argument("--run", action="store_true", help="also run verifier.py on every rollout and compare with labels.json (runs the submitted code)")
    sc.add_argument("--llm-base-url", help="OpenAI-compatible endpoint passed to the verifier as `llm` (key from OPENAI_API_KEY)")
    a = ap.parse_args(argv)
    if a.cmd == "check-submission":
        from .submission import check_submission
        problems, warnings, report = check_submission(a.path, run=a.run, llm_base_url=a.llm_base_url)
        print(json.dumps(report, indent=1))
        for p in problems:
            print("  PROBLEM:", p)
        for w in warnings:
            print("  warning:", w)
        if not problems:
            print("  OK")
        run = report.get("verifier_run") or {}
        return 1 if problems or run.get("errors") else 0
    bad = False
    for f in a.files:
        problems, warnings, info = check(f)
        print(f"{f}: {info.get('calls', 0)} calls  {json.dumps({k: v for k, v in info.items() if k != 'header' and k != 'calls'})}")
        for p in problems:
            print("  PROBLEM:", p); bad = True
        for w in warnings:
            print("  warning:", w)
        if not problems:
            print("  OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
