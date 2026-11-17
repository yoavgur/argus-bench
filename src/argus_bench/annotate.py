"""Add information to a rollout file after the fact (a human annotator's notes, a second grader's score, ...)."""
import os

from .check import _append, _read
import time


def annotate(path, data, call=None, message=None, by=None):
    """Append an annotation to the rollout file `path`. The original lines are never touched (the file stays append-only).

    data:    anything JSON-able (a dict such as {"misbehavior": True, "note": "deletes the backup"} is typical)
    call:    index of the model call it is about (0-based), or None for the whole rollout
    message: index inside that call's `messages` (0-based), if the note is about one input message; needs `call`
    by:      who wrote it ("annotator_2")

    Verifiers see it in rollout["calls"][call]["annotations"] (or rollout["annotations"] for the whole rollout). Monitors never see it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _, recs, _, _ = _read(path)
    if message is not None and call is None:
        raise ValueError("message= needs call=")
    if call is not None and not (isinstance(call, int) and 0 <= call < len(recs)):
        raise ValueError(f"call {call!r} does not exist: the file has {len(recs)} calls")
    if message is not None and not (isinstance(message, int) and 0 <= message < len(recs[call].get("messages") or [])):
        raise ValueError(f"message {message!r} does not exist in call {call}")
    _append(path, {"annotation": {"call": call, "message": message, "data": data, "by": by, "t": time.time()}})
