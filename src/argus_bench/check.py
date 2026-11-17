"""Structural check of a rollout file. No model needed."""
import json
import os


def load(path):
    return _read(path)[:2]


def _append(path, obj):
    """Append one JSON line with a single write() call (safe with several writers)."""
    fd = os.open(os.fspath(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _read(path):
    """(header, records, end, annotations): `end` is the last {"end": ...} line, the state declared when the run finished (or None);
    `annotations` are the {"annotation": ...} lines added after the fact, in file order."""
    header, recs, end, notes = {}, [], None, []
    for i, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: not valid JSON ({e})")
        if "annotation" in obj and "messages" not in obj:
            notes.append(obj["annotation"])
        elif "end" in obj and "messages" not in obj:
            end = obj["end"] if end is None or obj["end"].get("t", 0) >= end.get("t", 0) else end
        elif "header" in obj:
            h = dict(obj["header"])
            if isinstance(h.get("models"), dict) and isinstance(header.get("models"), dict):
                h["models"] = {**header["models"], **h["models"]}          # several processes may declare different models
            header.update(h)
        else:
            recs.append(obj)
    return header, recs, end, notes


RUN_FIELDS = ("revision", "dtype", "quantization", "chat_template")


def model_info(header, model):
    """How `model` was run: header['models'][model] if declared there, else the single-model fields at the top of the header
    (revision, dtype, quantization, chat_template)."""
    declared = (header.get("models") or {}).get(model)
    if declared is not None:
        return dict(declared)
    return {k: header[k] for k in RUN_FIELDS if header.get(k) is not None}


def _same(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def check(path):
    """Returns (problems, warnings, info)."""
    header, recs, _, notes = _read(path)
    problems, warnings, info = [], [], {}
    if not recs:
        return ["no model calls in file"], [], {}
    for n in notes:
        c, m = n.get("call"), n.get("message")
        if c is not None and not (isinstance(c, int) and 0 <= c < len(recs)):
            problems.append(f"annotation points to call {c!r}, but the file has {len(recs)} calls")
        elif m is not None and (c is None or not (isinstance(m, int) and 0 <= m < len(recs[c].get("messages") or []))):
            problems.append(f"annotation points to message {m!r} of call {c!r}, which does not exist")
    for i, r in enumerate(recs):
        w = f"call {i}"
        if not r.get("messages"):
            problems.append(f"{w}: no messages")
        out = r.get("output") or {}
        if not (out.get("content") or out.get("tool_calls") or out.get("reasoning_content")):
            problems.append(f"{w}: empty output")
        for tc in out.get("tool_calls") or []:
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    json.loads(a)
                except json.JSONDecodeError:
                    warnings.append(f"{w}: tool call '{tc.get('name')}' has arguments that are not valid JSON (kept as-is)")
    models = {r.get("model") for r in recs}
    if None in models or "" in models:
        warnings.append("some calls have no `model`; pass model=... to log()")
    if not header.get("library"):
        warnings.append("no `library` in the header (e.g. library='vllm 0.23.0 (server)')")
    names = sorted(m for m in models if m)
    declared = header.get("models") or {}
    for name in names:
        info = model_info(header, name)
        who = f"model {name!r}: " if len(names) > 1 or declared else ""
        if len(names) > 1 and name not in declared and not info:
            warnings.append(f"{who}not declared in the header `models` (declare revision, dtype, quantization, chat_template for every model used)")
            continue
        if not info.get("revision"):
            warnings.append(f"{who}no model revision declared: use record_to(path, revision='<full hf commit hash>') or models={{name: {{...}}}}")
        for k, hint in (("dtype", "bfloat16"), ("quantization", "none"), ("chat_template", "default")):
            if not info.get(k):
                warnings.append(f"{who}no `{k}` declared (e.g. {k}={hint!r})")
    for name in declared:
        if name not in names:
            warnings.append(f"header declares model {name!r} but no call used it")
    if not any(r.get("sampling") for r in recs):
        warnings.append("no sampling settings recorded (temperature/top_p/...)")
    # how the context evolved: extension of the previous call, or something else (e.g. compaction, a new thread)
    ext = other = 0
    prev = None
    for r in recs:
        if prev is not None:
            base = prev["messages"] + [{"role": "assistant"}]
            if len(r["messages"]) >= len(base) and all(_same(a, b) for a, b in zip(prev["messages"], r["messages"])):
                ext += 1
            else:
                other += 1
        prev = r
    info = {"calls": len(recs), "models": sorted(m for m in models if m), "calls_extending_previous": ext,
            "calls_with_new_context (compaction / new thread)": other, "annotations": len(notes), "header": header}
    return problems, warnings, info
