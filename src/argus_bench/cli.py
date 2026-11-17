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
    an = sub.add_parser("annotate", help="add a note to a rollout file after the fact (appends a line, never edits)")
    an.add_argument("file")
    an.add_argument("--call", type=int, help="model call index (0-based); omit for the whole rollout")
    an.add_argument("--message", type=int, help="message index inside that call")
    an.add_argument("--by", help="who is annotating")
    g = an.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="JSON value, or plain text")
    g.add_argument("--data-file", help="file with the JSON value")
    a = ap.parse_args(argv)
    if a.cmd == "annotate":
        from .annotate import annotate
        raw = a.data if a.data is not None else open(a.data_file).read()
        try:
            data = json.loads(raw)
        except ValueError:
            data = raw
        annotate(a.file, data, call=a.call, message=a.message, by=a.by)
        print("annotation added")
        return 0
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
        try:
            problems, warnings, info = check(f)
        except ValueError as e:
            print(f"{f}: PROBLEM: {e}")
            bad = True
            continue
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
