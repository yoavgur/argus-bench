import argparse
import json
import sys

from .check import check


def main(argv=None):
    ap = argparse.ArgumentParser(prog="argus-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check that a rollout file is well-formed")
    c.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
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
