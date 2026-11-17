"""Started by argus_bench.pth in every Python process whose environment has ARGUS_BENCH_LOG set."""
import json
import os


def start():
    path = os.environ.get("ARGUS_BENCH_LOG")
    if not path:
        return
    try:
        meta = json.loads(os.environ.get("ARGUS_BENCH_META") or "{}")
    except ValueError:
        meta = {}
    import argus_bench
    argus_bench.record_to(path, _quiet=True, **meta)
