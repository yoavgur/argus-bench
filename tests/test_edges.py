"""Edge cases and error paths."""
import asyncio
import json
import os
import sys
import textwrap

import httpx
import openai
import pytest

import argus_bench
from argus_bench import Span, Verdict, annotate, load_rollout, monitor_view
from argus_bench import _auto
from argus_bench.check import check, load
from argus_bench.cli import main
from argus_bench.submission import check_submission

MSGS = [{"role": "user", "content": "hi"}]
FULL = {"revision": "a" * 40, "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}


def chat(n=1):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": i, "message": {"role": "assistant", "content": f"answer {i}"}, "finish_reason": "stop"} for i in range(n)]}


def client(handler):
    return openai.OpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.fixture
def out(tmp_path):
    p = tmp_path / "r.jsonl"
    argus_bench.record_to(p, library="x", **FULL)
    yield p
    argus_bench.env_state(None)
    argus_bench.stop()


# ------------------------------------------------------------ what gets recorded and what does not
def test_other_endpoints_are_not_recorded_and_still_work(out):
    def handler(req):
        if req.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}], "model": "e", "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "m", "object": "model", "created": 0, "owned_by": "x"}]})
        return httpx.Response(200, json=chat())
    c = client(handler)
    assert c.embeddings.create(model="e", input="hi").data[0].embedding == [0.1, 0.2]
    assert c.models.list().data[0].id == "m"
    c.chat.completions.create(model="m", messages=MSGS)
    assert len(load(out)[1]) == 1                                          # only the chat call


def test_several_choices_give_one_record_each(out):
    client(lambda r: httpx.Response(200, json=chat(3))).chat.completions.create(model="m", messages=MSGS, n=3)
    recs = load(out)[1]
    assert [r["output"]["content"] for r in recs] == ["answer 0", "answer 1", "answer 2"]


def test_async_stream_as_context_manager_closed_early(out):
    def handler(req):
        chunks = "".join(f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm', 'choices': [{'index': 0, 'delta': {'content': w}, 'finish_reason': None}]})}\n\n" for w in ("a", "b", "c", "d"))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks + "data: [DONE]\n\n")

    async def go():
        c = openai.AsyncOpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        async with await c.chat.completions.create(model="m", messages=MSGS, stream=True) as s:
            async for i, ch in aiter_enumerate(s):
                if i == 1:
                    break
    asyncio.run(go())
    assert load(out)[1][0]["output"]["content"].startswith("ab")


async def aiter_enumerate(s):
    i = 0
    async for x in s:
        yield i, x
        i += 1


def test_unserialisable_values_do_not_crash_the_agent(out):
    class Odd:
        def __str__(self):
            return "odd-object"
    argus_bench.log([{"role": "user", "content": "x", "extra": Odd()}], {"content": "ok", "tool_calls": []}, model="m", marker=Odd())
    r = load(out)[1][0]
    assert r["messages"][0]["extra"] == "odd-object" and r["marker"] == "odd-object" and r["output"] == {"content": "ok"}


def test_pydantic_objects_are_dumped(out):
    msg = openai.types.chat.ChatCompletionMessage(role="assistant", content="from pydantic")
    argus_bench.log([MSGS[0], msg], "ok", model="m")
    assert load(out)[1][0]["messages"][1]["content"] == "from pydantic"


def test_extra_body_sampling_fields_are_recorded(out):
    client(lambda r: httpx.Response(200, json=chat())).chat.completions.create(model="m", messages=MSGS, extra_body={"top_k": 7, "min_p": 0.1, "not_sampling": 1})
    assert load(out)[1][0]["sampling"] == {"top_k": 7, "min_p": 0.1}


# ------------------------------------------------------------ the hook
def test_warns_when_the_sdk_internals_changed(tmp_path, monkeypatch):
    import openai._base_client as bc
    monkeypatch.setattr(bc.SyncAPIClient, "request", None)                  # simulate a future SDK without this function
    with pytest.warns(UserWarning, match="cannot hook openai"):
        argus_bench.record_to(tmp_path / "w.jsonl", library="x")
    argus_bench.stop()


def test_works_without_the_openai_package(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)                       # `import openai` raises ImportError
    p = tmp_path / "n.jsonl"
    argus_bench.record_to(p, library="x", **FULL)
    argus_bench.log(MSGS, "ok", model="m", sampling={"temperature": 1})
    argus_bench.stop()
    h, recs = load(p)
    assert "openai_sdk" not in h and len(recs) == 1


def test_stop_really_stops_recording(out):
    argus_bench.stop()
    argus_bench.log(MSGS, "ignored", model="m")
    client(lambda r: httpx.Response(200, json=chat())).chat.completions.create(model="m", messages=MSGS)
    assert not out.exists()


# ------------------------------------------------------------ env_state and end
def test_end_without_state_or_declaration_writes_nothing(out):
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end()
    assert load_rollout(out)["end"] is None


def test_end_evaluates_the_declared_function_and_exit_hook_does_not_repeat_it(out):
    argus_bench.env_state(lambda: {"final": 1})
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end()
    argus_bench._end_at_exit()                                              # what runs at interpreter exit: must not write a second end
    assert out.read_text().count('"end"') == 1 and load_rollout(out)["end"]["env_state"] == {"final": 1}


def test_exit_hook_writes_final_state_when_end_was_never_called(out):
    argus_bench.env_state(lambda: {"final": 2})
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench._end_at_exit()
    assert load_rollout(out)["end"]["env_state"] == {"final": 2}


def test_exit_hook_stays_quiet_if_no_calls_were_made(out):
    argus_bench.env_state(lambda: {"final": 3})
    argus_bench._end_at_exit()
    assert not out.exists()


def test_env_var_activation_in_process(tmp_path, monkeypatch):
    p = tmp_path / "auto.jsonl"
    monkeypatch.setenv("ARGUS_BENCH_LOG", str(p))
    monkeypatch.setenv("ARGUS_BENCH_META", json.dumps({"library": "from-env", "revision": "b" * 40}))
    _auto.start()
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.stop()
    assert load(p)[0]["library"] == "from-env"
    monkeypatch.setenv("ARGUS_BENCH_META", "{not json")                      # a broken META must not break the agent
    _auto.start()
    argus_bench.stop()
    monkeypatch.delenv("ARGUS_BENCH_LOG")
    _auto.start()                                                            # no variable: nothing happens
    assert argus_bench._S["path"] is None


# ------------------------------------------------------------ check
def test_check_rejects_files_that_are_not_jsonl(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"messages": []}\nthis is not json\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        check(p)


def test_check_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    assert check(p)[0] == ["no model calls in file"]


def test_check_model_declaration_warnings(tmp_path):
    p = tmp_path / "m.jsonl"
    argus_bench.record_to(p, library="x", models={"a": {"dtype": "bfloat16", "quantization": "none", "chat_template": "default"}, "unused": FULL})
    argus_bench.log(MSGS, "one", model="a", sampling={"temperature": 1})
    argus_bench.stop()
    w = " | ".join(check(p)[1])
    assert "no model revision declared" in w and "'unused' but no call used it" in w


def test_check_flags_annotation_on_a_message_that_does_not_exist(tmp_path):
    p = tmp_path / "a.jsonl"
    argus_bench.record_to(p, library="x", **FULL)
    argus_bench.log(MSGS, "ok", model="m", sampling={"temperature": 1})
    argus_bench.stop()
    with open(p, "a") as f:
        f.write(json.dumps({"annotation": {"call": 0, "message": 9, "data": "x"}}) + "\n")
    assert any("message 9" in x for x in check(p)[0])


# ------------------------------------------------------------ rollouts and views
def test_load_rollout_ignores_an_annotation_pointing_nowhere_and_handles_sparse_records(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"messages": MSGS, "output": {"content": "x"}}) + "\n" + json.dumps({"annotation": {"call": 5, "data": "lost"}}) + "\n")
    r = load_rollout(p)
    assert r["calls"][0]["annotations"] == [] and r["annotations"] == [] and r["end"] is None
    assert monitor_view(r) == {"calls": [{"messages": MSGS, "output": {"content": "x"}}]}


def test_verdict_from_dict_with_spans():
    v = Verdict.from_dict({"misbehavior": 1, "explanation": "e", "spans": [{"call": 1, "location": "output.content", "start": 2, "end": 4, "text": "ab"}]})
    assert v.misbehavior is True and v.spans == [Span(1, "output.content", 2, 4, "ab")]


# ------------------------------------------------------------ the command line, in-process
def test_cli_check_and_annotate_and_check_submission(tmp_path, capsys, server):
    p = tmp_path / "c.jsonl"
    argus_bench.record_to(p, library="x", **FULL)
    argus_bench.log(MSGS, "ok", model="m", sampling={"temperature": 1})
    argus_bench.stop()
    assert main(["check", str(p)]) == 0 and "OK" in capsys.readouterr().out
    assert main(["annotate", str(p), "--call", "0", "--data", "plain words"]) == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"messages": [], "output": {}}) + "\n")
    assert main(["check", str(bad)]) == 1 and "PROBLEM" in capsys.readouterr().out
    sub = tmp_path / "sub"
    (sub / "rollouts").mkdir(parents=True)
    (sub / "rollouts" / "a.jsonl").write_text(p.read_text())
    (sub / "labels.json").write_text(json.dumps({"a.jsonl": True}))
    (sub / "verifier.py").write_text("from argus_bench import Verdict\ndef verify(rollout, artifacts_dir, llm):\n    return Verdict(True, 'x')\n")
    (sub / "verifier-requirements.txt").write_text("")
    assert main(["check-submission", str(sub), "--run"]) == 0
    assert main(["check-submission", str(sub), "--run", "--llm-base-url", server]) == 0     # an llm client is built and passed


# ------------------------------------------------------------ submissions: every way to get the layout wrong
def make(root, **over):
    files = {"labels.json": json.dumps({"a.jsonl": True, "b.jsonl": False}),
             "verifier.py": "from argus_bench import Verdict\ndef verify(rollout, artifacts_dir, llm):\n    return Verdict(False)\n",
             "verifier-requirements.txt": ""}
    files.update(over)
    (root / "rollouts").mkdir(parents=True)
    for n in ("a.jsonl", "b.jsonl"):
        p = root / "rollouts" / n
        argus_bench.record_to(p, library="x", **FULL)
        argus_bench.log(MSGS, "ok", model="m", sampling={"temperature": 1})
        argus_bench.stop()
    for name, content in files.items():
        if content is not None:
            (root / name).write_text(content)


@pytest.mark.parametrize("over,expect", [
    ({"labels.json": None}, "labels.json is missing"),
    ({"labels.json": "{oops"}, "labels.json is not valid JSON"),
    ({"verifier.py": None}, "verifier.py is missing"),
    ({"verifier.py": "x = 1\n"}, "does not define"),
    ({"verifier-requirements.txt": None}, "verifier-requirements.txt is missing"),
    ({"verifier_config.json": "{oops"}, "verifier_config.json is not valid JSON"),
    ({"verifier_config.json": json.dumps({"model": "m"})}, 'needs "revision"'),
])
def test_submission_problems(tmp_path, over, expect):
    make(tmp_path, **over)
    problems, _, _ = check_submission(tmp_path)
    assert any(expect in p for p in problems), problems


def test_submission_warnings(tmp_path):
    make(tmp_path, **{"labels.json": json.dumps({"a.jsonl": True, "b.jsonl": True}), "verifier_config.json": json.dumps({"model": "m", "revision": "short"})})
    (tmp_path / "rollouts" / "notes.txt").write_text("not a rollout")
    _, warnings, _ = check_submission(tmp_path)
    text = " | ".join(warnings)
    assert "only one kind of label" in text and "revision should be the full 40-character" in text and "not a .jsonl file, ignored" in text


def test_submission_with_two_models_warns(tmp_path):
    make(tmp_path)
    p = tmp_path / "rollouts" / "b.jsonl"
    argus_bench.record_to(p, library="x", models={"m": FULL, "other": FULL})
    argus_bench.log(MSGS, "ok", model="other", sampling={"temperature": 1})
    argus_bench.stop()
    assert any("several models" in w for w in check_submission(tmp_path)[1])


def test_verifier_using_llm_without_an_endpoint_gets_none_and_a_warning(tmp_path):
    make(tmp_path, **{"verifier.py": "from argus_bench import Verdict\ndef verify(rollout, artifacts_dir, llm):\n    return Verdict(llm is None)\n",
                      "verifier_config.json": json.dumps({"model": "m", "revision": "a" * 40})})
    problems, warnings, report = check_submission(tmp_path, run=True, log=lambda *_: None)
    assert not problems and any("llm=None" in w for w in warnings) and report["verifier_run"]["accuracy"] == 0.5


def test_unreadable_rollout_in_a_submission_is_a_problem_not_a_crash(tmp_path):
    make(tmp_path)
    (tmp_path / "rollouts" / "a.jsonl").write_text("not json at all\n")
    problems, _, _ = check_submission(tmp_path)
    assert any("rollouts/a.jsonl" in p and "not valid JSON" in p for p in problems)


def test_cli_check_reports_unreadable_file_instead_of_a_traceback(tmp_path, capsys):
    p = tmp_path / "x.jsonl"
    p.write_text("garbage\n")
    assert main(["check", str(p)]) == 1 and "PROBLEM" in capsys.readouterr().out


def test_explicit_close_of_streams(out):
    def handler(req):
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [{"index": 0, "delta": {"content": "hey"}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
    s = client(handler).chat.completions.create(model="m", messages=MSGS, stream=True)
    next(iter(s))
    s.close()

    async def go():
        c = openai.AsyncOpenAI(base_url="http://x/v1", api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        a = await c.chat.completions.create(model="m", messages=MSGS, stream=True)
        async for _ in a:
            break
        await a.close()
    asyncio.run(go())
    recs = load(out)[1]
    assert len(recs) == 2 and all(r["output"]["content"] == "hey" for r in recs)


def test_check_warns_when_the_run_ends_right_after_a_tool_call(tmp_path):
    p = tmp_path / "t.jsonl"
    argus_bench.record_to(p, library="x", **FULL)
    argus_bench.log(MSGS, {"content": "", "tool_calls": [{"id": "1", "name": "bash", "arguments": "{}"}]}, model="m", sampling={"temperature": 1})
    argus_bench.stop()
    assert any("ends right after a tool call" in w for w in check(p)[1])
    q = tmp_path / "u.jsonl"                                                  # a run that ends with a final answer: no warning
    argus_bench.record_to(q, library="x", **FULL)
    argus_bench.log(MSGS, "final answer", model="m", sampling={"temperature": 1})
    argus_bench.stop()
    assert not any("tool call" in w for w in check(q)[1])


def test_stop_writes_the_final_state_if_end_was_not_called(tmp_path):
    p = tmp_path / "s.jsonl"
    argus_bench.record_to(p, library="x")
    argus_bench.env_state(lambda: {"final": "state"})
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.stop()
    argus_bench._end_at_exit()                                              # the interpreter-exit hook afterwards must not add another
    argus_bench.env_state(None)
    assert p.read_text().count('"end"') == 1 and load_rollout(p)["end"]["env_state"] == {"final": "state"}
    q = tmp_path / "t.jsonl"                                                # an explicit end() is not repeated by stop()
    argus_bench.record_to(q, library="x")
    argus_bench.env_state(lambda: {"final": "auto"})
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end({"final": "explicit"})
    argus_bench.stop()
    argus_bench.env_state(None)
    assert q.read_text().count('"end"') == 1 and load_rollout(q)["end"]["env_state"] == {"final": "explicit"}
