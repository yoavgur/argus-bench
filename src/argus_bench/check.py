"""Structural check of a rollout file. No model needed."""
import json


def load(path):
    header, recs = {}, []
    for i, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: not valid JSON ({e})")
        if "header" in obj:
            header.update(obj["header"])
        else:
            recs.append(obj)
    return header, recs


def _same(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def check(path):
    """Returns (problems, warnings, info)."""
    header, recs = load(path)
    problems, warnings, info = [], [], {}
    if not recs:
        return ["no model calls in file"], [], {}
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
    if not header.get("revision"):
        warnings.append("no model revision in the header: use record_to(path, revision='<full hf commit hash>')")
    if not header.get("library"):
        warnings.append("no `library` in the header (e.g. library='vllm 0.23.0 (server)')")
    if not header.get("dtype"):
        warnings.append("no `dtype` in the header (e.g. dtype='bfloat16')")
    if not header.get("quantization"):
        warnings.append("no `quantization` declared in the header (e.g. quantization='none')")
    if not header.get("chat_template"):
        warnings.append("no `chat_template` declared in the header (e.g. chat_template='default')")
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
            "calls_with_new_context (compaction / new thread)": other, "header": header}
    return problems, warnings, info
