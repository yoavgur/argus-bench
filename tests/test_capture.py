import asyncio
import inspect
import json

import httpx
import openai
import pytest

import argus_bench
from argus_bench.check import check, load

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object", "properties": {}}}}]


def completion(content="hello", reasoning=None, calls=None):
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps(calls)}}]
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}]}


def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def chunk(delta, finish=None):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def handler(request):
    body = json.loads(request.content)
    if body.get("stream"):
        chunks = [chunk({"role": "assistant", "reasoning_content": "th"}), chunk({"reasoning_content": "ink"}), chunk({"content": "hel"}),
                  chunk({"content": "lo"}), chunk({"tool_calls": [{"index": 0, "id": "c9", "type": "function", "function": {"name": "bash", "arguments": '{"command"'}}]}),
                  chunk({"tool_calls": [{"index": 0, "function": {"arguments": ': "ls"}'}}]}, "tool_calls")]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(chunks))
    return httpx.Response(200, json=completion("hello", reasoning="think", calls={"command": "ls"}))


@pytest.fixture
def out(tmp_path):
    p = tmp_path / "r.jsonl"
    argus_bench.record_to(p, revision="abc")
    yield p
    argus_bench.stop()


def test_sync(out):
    c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    r = c.chat.completions.create(model="m", messages=MSGS, tools=TOOLS, temperature=0.7, seed=1, extra_body={"top_k": 20})
    assert r.choices[0].message.content == "hello"                       # the user still gets the normal response
    header, recs = load(out)
    assert header["revision"] == "abc" and len(recs) == 1
    rec = recs[0]
    assert rec["messages"] == MSGS and rec["tools"] == TOOLS and rec["model"] == "m"
    assert rec["sampling"] == {"temperature": 0.7, "seed": 1, "top_k": 20}
    assert rec["output"]["reasoning_content"] == "think" and rec["output"]["tool_calls"][0]["name"] == "bash"


def test_stream_sync(out):
    c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    seen = [ch for ch in c.chat.completions.create(model="m", messages=MSGS, stream=True)]
    assert len(seen) == 6                                                # all chunks still reach the caller
    _, recs = load(out)
    o = recs[0]["output"]
    assert o["content"] == "hello" and o["reasoning_content"] == "think" and recs[0]["stream"] is True
    assert o["tool_calls"] == [{"id": "c9", "name": "bash", "arguments": '{"command": "ls"}'}]


def test_stream_context_manager_and_early_close(out):
    c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with c.chat.completions.create(model="m", messages=MSGS, stream=True) as s:
        for i, ch in enumerate(s):
            if i == 3:
                break
    _, recs = load(out)
    assert len(recs) == 1 and recs[0]["output"]["content"].startswith("hel")      # partial output is still logged


def test_async_and_async_stream(out):
    async def go():
        c = openai.AsyncOpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        await c.chat.completions.create(model="m", messages=MSGS)
        s = await c.chat.completions.create(model="m", messages=MSGS, stream=True)
        return [ch async for ch in s]
    assert len(asyncio.run(go())) == 6
    _, recs = load(out)
    assert len(recs) == 2 and recs[1]["output"]["content"] == "hello" and recs[1]["stream"]


def test_log_primitive_and_check(tmp_path):
    p = tmp_path / "x.jsonl"
    argus_bench.record_to(p, revision="abc")
    argus_bench.log(MSGS, "hello", model="m", sampling={"temperature": 0.6})
    argus_bench.log(MSGS + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}], {"content": "ok", "tool_calls": [{"name": "bash", "arguments": "{bad"}]}, model="m")
    argus_bench.log([MSGS[0], {"role": "user", "content": "summary of the conversation"}], "fresh", model="m")     # compaction
    argus_bench.stop()
    problems, warnings, info = check(p)
    assert not problems
    assert any("not valid JSON" in w for w in warnings)
    assert info["calls_extending_previous"] == 1 and info["calls_with_new_context (compaction / new thread)"] == 1


def test_check_flags_empty_and_missing(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"messages": [], "output": {}}) + "\n")
    problems, _, _ = check(p)
    assert any("no messages" in x for x in problems) and any("empty output" in x for x in problems)


def test_patch_is_idempotent(tmp_path):
    argus_bench.record_to(tmp_path / "a.jsonl")
    argus_bench.record_to(tmp_path / "b.jsonl")
    c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    c.chat.completions.create(model="m", messages=MSGS)
    argus_bench.stop()
    assert len(load(tmp_path / "b.jsonl")[1]) == 1 and not (tmp_path / "a.jsonl").exists()


def test_raw_response_path_used_by_litellm_and_langchain(out):
    c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    raw = c.chat.completions.with_raw_response.create(model="m", messages=MSGS)
    r = raw.parse()
    assert r.choices[0].message.content == "hello" and raw.parse() is r          # same object on repeated parse, logged once
    s = c.chat.completions.with_raw_response.create(model="m", messages=MSGS, stream=True).parse()
    assert len([ch for ch in s]) == 6
    _, recs = load(out)
    assert len(recs) == 2 and recs[1]["stream"] and recs[1]["output"]["content"] == "hello"


def test_raw_response_async(out):
    async def go():
        c = openai.AsyncOpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        async def parsed(raw):
            r = raw.parse()
            return await r if inspect.isawaitable(r) else r
        raw = await c.chat.completions.with_raw_response.create(model="m", messages=MSGS)
        await parsed(raw)
        raw2 = await c.chat.completions.with_raw_response.create(model="m", messages=MSGS, stream=True)
        return [ch async for ch in await parsed(raw2)]
    assert len(asyncio.run(go())) == 6
    assert len(load(out)[1]) == 2


def test_env_state_function_evaluated_per_call(out):
    c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    state = {"n": 0}
    argus_bench.env_state(lambda: {"files": state["n"]})
    c.chat.completions.create(model="m", messages=MSGS)
    state["n"] = 3
    c.chat.completions.create(model="m", messages=MSGS, stream=True).__next__()
    argus_bench.log(MSGS, "x", model="m")
    argus_bench.log(MSGS, "y", model="m", env_state="explicit")
    argus_bench.env_state(None)
    c.chat.completions.create(model="m", messages=MSGS)
    _, recs = load(out)
    assert [r.get("env_state") for r in recs] == [{"files": 0}, {"files": 3}, {"files": 3}, "explicit", None]
    assert "__env_state" not in json.dumps(recs)


def test_warns_when_nothing_recorded(tmp_path):
    argus_bench.record_to(tmp_path / "e.jsonl")
    with pytest.warns(UserWarning, match="no model call was recorded"):
        argus_bench._no_calls_warning()
    argus_bench.stop()


def test_check_warns_on_undeclared_quantization_and_template(tmp_path):
    p = tmp_path / "q.jsonl"
    argus_bench.record_to(p, revision="abc")
    argus_bench.log(MSGS, "hello", model="m", sampling={"temperature": 1})
    argus_bench.stop()
    _, warnings_, _ = check(p)
    assert any("quantization" in w for w in warnings_) and any("chat_template" in w for w in warnings_)


def test_subprocess_agent_records_via_env_var_without_code_changes(tmp_path):
    import os, subprocess, sys, textwrap
    script = tmp_path / "agent.py"          # an "agent" that knows nothing about argus_bench
    script.write_text(textwrap.dedent("""
        import httpx, json, openai
        h = lambda r: httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi from child"}, "finish_reason": "stop"}]})
        c = openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(h)))
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    """))
    log = tmp_path / "sub.jsonl"
    env = {**os.environ, "ARGUS_BENCH_LOG": str(log), "ARGUS_BENCH_META": json.dumps({"revision": "r1", "quantization": "none"})}
    subprocess.run([sys.executable, str(script)], env=env, check=True)
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)           # a process that makes no call leaves nothing behind
    header, recs = load(log)
    assert header["revision"] == "r1" and len(recs) == 1 and recs[0]["output"]["content"] == "hi from child"


def test_many_processes_do_not_interleave_lines(tmp_path):
    import subprocess, sys, textwrap, os
    log = tmp_path / "many.jsonl"
    code = textwrap.dedent("""
        import argus_bench, sys
        argus_bench.record_to(sys.argv[1], _quiet=True)
        for i in range(30):
            argus_bench.log([{"role": "user", "content": "x" * 20000}], "y" * 20000, model="m")
    """)
    ps = [subprocess.Popen([sys.executable, "-c", code, str(log)]) for _ in range(4)]
    assert all(p.wait() == 0 for p in ps)
    _, recs = load(log)                                  # load() raises on any torn line
    assert len(recs) == 120
