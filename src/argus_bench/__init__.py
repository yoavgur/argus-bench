"""argus_bench: write down what a model was shown and what it said, one JSON line per model call.

    import argus_bench
    argus_bench.record_to("rollout.jsonl")     # every OpenAI-SDK chat call from here on is logged (sync, async, streaming)

    argus_bench.log(messages, output, tools=tools, model="Qwen/Qwen3-1.7B")   # anything else: one call, two values

No model, library or server specific flags. Nothing to configure per library. Check the file with `argus-bench check rollout.jsonl`.
If the agent runs in another process or container and you cannot edit its code, set ARGUS_BENCH_LOG=/path/rollout.jsonl in its environment instead
(optionally ARGUS_BENCH_META='{"revision": "..."}'); the installed package then starts recording in every Python process there.
"""
import functools
import inspect
import json
import os
import platform
import threading
import time
import warnings
import atexit

__version__ = "0.1.1"

from .types import Span, Verdict                      # noqa: E402
from .rollout import load_rollout, monitor_view      # noqa: E402
__all__ = ["record_to", "log", "env_state", "end", "stop", "Verdict", "Span", "load_rollout", "monitor_view"]

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "max_completion_tokens", "seed", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "stop")
_S = {"path": None, "lock": threading.Lock(), "meta": {}, "n": 0, "env": None, "hooked": False, "ended": False}


# ------------------------------------------------------------------ writing
def _jsonable(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _write(rec):
    if not _S["path"]:
        return
    if "end" not in rec:
        _S["n"] += 1
    with _S["lock"]:
        h = _S.pop("header", None)
        data = (json.dumps(_jsonable(h), ensure_ascii=False) + "\n" if h else "") + json.dumps(_jsonable(rec), ensure_ascii=False) + "\n"
        fd = os.open(_S["path"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)       # one write() call: lines from several processes never interleave
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)


def record_to(path, _quiet=False, **meta):
    """Start appending one record per model call to `path`. `meta` goes in the header. Declare what you can:
    rollout_id="...", library="vllm 0.23", and how the model was run:
        revision="<full hf commit hash>", dtype="bfloat16", quantization="none" | "gguf-q4_k_m" | ..., chat_template="default" | "<name or hash>"
    If the run uses several models (e.g. a second model that summarises the context), declare each one by name instead:
        models={"Qwen/Qwen3-1.7B": {"revision": "...", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}, "Qwen/Qwen3-0.6B": {...}}"""
    _S["path"], _S["meta"], _S["n"], _S["quiet"], _S["ended"] = os.fspath(path), meta, 0, _quiet, False
    header = {"header": {"argus_bench": __version__, "python": platform.python_version(), "started": time.time(), **meta}}
    try:
        import openai
        header["header"]["openai_sdk"] = openai.__version__
    except ImportError:
        pass
    _S["header"] = header                       # written together with the first record, so processes that make no model call leave no trace
    _patch_openai()


def stop():
    _S["path"] = None


def env_state(state):
    """Declare the environment state to attach to every following call, as a JSON-able value or a zero-argument function.
    A function is called at the START of each model call (so state that changes between calls is captured when the OpenAI hook is used);
    with log() it is called when you log. Pass None to stop. Keep it small: put big state in a file and record its path or hash here."""
    _S["env"] = state


def end(state=None):
    """Declare the state of the environment at the END of the run (final file contents, command log, answer key, labels from annotators, ...).
    `state` is JSON-able; if omitted and a function was declared with env_state(), it is evaluated now. The verifier receives it as rollout["end"]["env_state"];
    monitors never see it. If you never call end() and declared env_state(), the final state is written automatically when the program exits.
    Keep it small: put big files in artifacts/<rollout id>/ and record paths or hashes here."""
    if state is None:
        state, on = _env()
        if not on:
            return
    _write({"end": {"t": time.time(), "env_state": state}})
    _S["ended"] = True


def _end_at_exit():
    if _S["path"] and _S["env"] is not None and not _S["ended"] and _S["n"] > 0:
        end()


atexit.register(_end_at_exit)


def _env():
    e = _S["env"]
    if e is None:
        return None, False
    try:
        return (e() if callable(e) else e), True
    except Exception as ex:
        return f"env_state error: {ex!r}", True


def _no_calls_warning():
    if _S["path"] and _S["n"] == 0 and not _S.get("quiet"):
        warnings.warn(f"argus_bench: record_to({_S['path']!r}) was called but no model call was recorded. Is the model called through the "
                      "openai Python SDK /chat/completions in this process? Otherwise call argus_bench.log(...) yourself.", stacklevel=1)


atexit.register(_no_calls_warning)


def _norm_output(output):
    if isinstance(output, str):
        return {"content": output}
    out = dict(output)
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def log(messages, output, tools=None, model=None, sampling=None, **extra):
    """Log one model call. `messages`: the chat messages the model was given (OpenAI format). `output`: the generated text, or a dict with
    content / reasoning_content / tool_calls. `sampling`: temperature, top_p, ... if you know them. Without a prior record_to(), call
    argus_bench.record_to(path) first. An `env_state=` keyword overrides the value set with argus_bench.env_state()."""
    if "env_state" not in extra:
        v, on = _env()
        if on:
            extra["env_state"] = v
    _write({"t": time.time(), "model": model, "messages": messages, "tools": tools, "sampling": sampling or {},
            "output": _norm_output(output), **extra})


# ------------------------------------------------------------------ OpenAI SDK hook
def _request_fields(kw):
    s = {k: kw[k] for k in SAMPLING_KEYS if kw.get(k) is not None}
    for k, v in (kw.get("extra_body") or {}).items():
        if k in SAMPLING_KEYS:
            s[k] = v
    rec = {"model": kw.get("model"), "messages": kw.get("messages"), "tools": kw.get("tools") or None, "sampling": s}
    if "__env_state" in kw:
        rec["env_state"] = kw["__env_state"]
    return rec


def _msg_output(m):
    extra = getattr(m, "model_extra", None) or {}
    out = {"content": getattr(m, "content", None), "reasoning_content": extra.get("reasoning_content") or extra.get("reasoning")}
    calls = getattr(m, "tool_calls", None)
    if calls:
        out["tool_calls"] = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments} for c in calls]
    return _norm_output(out)


def _log_completion(kw, res):
    for ch in getattr(res, "choices", None) or []:
        _write({"t": time.time(), **_request_fields(kw), "output": _msg_output(ch.message), "finish_reason": ch.finish_reason,
                "source": "openai-sdk"})


class _Acc:
    """Rebuilds the final message from streamed chunks."""

    def __init__(self, kw):
        self.kw, self.content, self.reasoning, self.calls, self.finish, self.done = kw, [], [], {}, None, False

    def add(self, chunk):
        for ch in getattr(chunk, "choices", None) or []:
            d = ch.delta
            extra = getattr(d, "model_extra", None) or {}
            if d.content:
                self.content.append(d.content)
            r = extra.get("reasoning_content") or extra.get("reasoning")
            if r:
                self.reasoning.append(r)
            for tc in d.tool_calls or []:
                c = self.calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.id or c["id"]
                if tc.function:
                    c["name"] += tc.function.name or ""
                    c["arguments"] += tc.function.arguments or ""
            self.finish = ch.finish_reason or self.finish

    def flush(self):
        if self.done:
            return
        self.done = True
        out = _norm_output({"content": "".join(self.content), "reasoning_content": "".join(self.reasoning),
                            "tool_calls": [self.calls[i] for i in sorted(self.calls)]})
        _write({"t": time.time(), **_request_fields(self.kw), "output": out, "finish_reason": self.finish, "source": "openai-sdk", "stream": True})


class _TeeStream:
    """Behaves like the SDK's Stream: iterable, an iterator (LiteLLM calls next()), a context manager, closable."""

    def __init__(self, stream, kw):
        self._s, self._it, self._a = stream, None, _Acc(kw)

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._s)
        try:
            chunk = next(self._it)
        except BaseException:
            self._a.flush()
            raise
        self._a.add(chunk)
        return chunk

    def __enter__(self):
        self._s.__enter__()
        return self

    def __exit__(self, *a):
        self._a.flush()
        return self._s.__exit__(*a)

    def close(self):
        self._a.flush()
        return self._s.close()

    def __del__(self):
        try:
            self._a.flush()          # consumer stopped early without closing: still log what the model produced
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._s, name)


class _TeeAsyncStream(_TeeStream):
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._s.__aiter__()
        try:
            chunk = await self._it.__anext__()
        except BaseException:
            self._a.flush()
            raise
        self._a.add(chunk)
        return chunk

    async def __aenter__(self):
        await self._s.__aenter__()
        return self

    async def __aexit__(self, *a):
        self._a.flush()
        return await self._s.__aexit__(*a)

    async def close(self):
        self._a.flush()
        return await self._s.close()


def _handle(kw, res):
    """Log a parsed result (ChatCompletion or Stream) and return what the caller should get back."""
    if kw.get("stream"):
        return (_TeeAsyncStream if hasattr(res, "__aiter__") else _TeeStream)(res, kw)
    _log_completion(kw, res)
    return res


def _wrap_result(body, res, is_async):
    """Log the parsed result (ChatCompletion or Stream); handle `with_raw_response`, where the parsed object only exists after .parse()."""
    if hasattr(res, "parse") and not hasattr(res, "choices"):
        parse, cache = res.parse, []

        def parse_logged(*a, **k):
            r = parse(*a, **k)
            if inspect.isawaitable(r):
                async def finish():
                    if not cache:
                        cache.append(_handle(body, await r))
                    return cache[0]
                return finish()
            if not cache:
                cache.append(_handle(body, r))
            return cache[0]
        res.parse = parse_logged
        return res
    return _handle(body, res)


def _chat_body(args, kw):
    options = kw.get("options") or (args[1] if len(args) > 1 else None)
    url = str(getattr(options, "url", "") or "")
    body = getattr(options, "json_data", None)
    if not (url.rstrip("/").endswith("/chat/completions") and isinstance(body, dict)):
        return None
    return {**body, **(getattr(options, "extra_json", None) or {})}          # extra_body fields (top_k, min_p, ...) are merged into the request


def _patch_openai():
    """Patch the SDK's single HTTP entry point, so every way of calling chat completions is covered: client.chat.completions.create,
    with_raw_response, client.post(...) (LiteLLM), LangChain, the Agents SDK's chat-completions model, sync / async / streaming."""
    try:
        import openai
    except ImportError:
        return                                   # no openai SDK: only log() is available
    try:
        from openai._base_client import AsyncAPIClient, SyncAPIClient
        assert callable(SyncAPIClient.request) and callable(AsyncAPIClient.request)
    except Exception:
        warnings.warn(f"argus_bench: cannot hook openai {getattr(openai, '__version__', '?')} (its internals changed). "
                      "Calls through the SDK will NOT be recorded; use argus_bench.log(...) instead.", stacklevel=3)
        return
    if getattr(SyncAPIClient.request, "_argus_bench", False):
        return
    sync_request, async_request = SyncAPIClient.request, AsyncAPIClient.request

    @functools.wraps(sync_request)
    def request(self, *args, **kw):
        body = _chat_body(args, kw)
        if body is not None:
            v, on = _env()
            if on:
                body["__env_state"] = v
        res = sync_request(self, *args, **kw)
        return _wrap_result(body, res, False) if body is not None else res

    @functools.wraps(async_request)
    async def arequest(self, *args, **kw):
        body = _chat_body(args, kw)
        if body is not None:
            v, on = _env()
            if on:
                body["__env_state"] = v
        res = await async_request(self, *args, **kw)
        return _wrap_result(body, res, True) if body is not None else res

    request._argus_bench = arequest._argus_bench = True
    SyncAPIClient.request, AsyncAPIClient.request = request, arequest
