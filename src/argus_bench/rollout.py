"""Loading a rollout file the way a verifier or a monitor receives it."""
import os

from .check import _read


def load_rollout(path):
    """The dict a verifier receives: everything recorded, and no label (labels live in labels.json, never in the rollout).

        {"id": "<file name without extension>",
         "header": {...run facts: library, models, rollout_id...},
         "calls": [{"model", "messages", "tools", "sampling", "output", "finish_reason", "env_state", ...}, ...],
         "end": {"t": ..., "env_state": ...} or None}      # state declared when the run finished, see argus_bench.end()
    """
    header, calls, end = _read(path)
    return {"id": os.path.splitext(os.path.basename(os.fspath(path)))[0], "header": header, "calls": calls, "end": end}


MONITOR_CALL_FIELDS = ("messages", "tools", "output", "finish_reason")


def monitor_view(rollout):
    """What a monitor may see: the transcript only. Hidden: env_state, the end state, the run header (rollout_id, library, revision, ...),
    sampling settings, timestamps and the rollout id. Use this function to build the monitor's input so that evidence meant only for the
    verifier cannot leak."""
    return {"calls": [{k: c[k] for k in MONITOR_CALL_FIELDS if k in c} for c in rollout["calls"]]}
