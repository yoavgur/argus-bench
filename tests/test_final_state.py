"""Everything about the end of a run: end(), stop(), the exit hook, env_state timing, tool loops that stop early, and what verifiers/monitors see."""
import json
import os
import random
import string
import subprocess
import sys
import textwrap
import threading

import openai
import pytest

import argus_bench
from argus_bench import annotate, load_rollout, monitor_view
from argus_bench.check import check
from argus_bench.submission import check_submission

MSGS = [{"role": "user", "content": "hi"}]
FULL = {"revision": "a" * 40, "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}


@pytest.fixture(autouse=True)
def clean_state():
    yield
    argus_bench.env_state(None)
    argus_bench.stop()


def start(path, **meta):
    argus_bench.record_to(path, library="x", **{**FULL, **meta})


def py(code, **env):
    """Run `code` as its own program, as an agent process would."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env={**os.environ, **env}, capture_output=True, text=True)


def lines(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def ends(p):
    return [o for o in lines(p) if "end" in o]


# ===================================================================== how the run ends: five ways
RECORD = """import argus_bench
argus_bench.record_to({path!r}, library="x")
argus_bench.env_state(lambda: {{"final": "state"}})
argus_bench.log([{{"role": "user", "content": "hi"}}], "ok", model="m")
"""


@pytest.mark.parametrize("ending,expected_end_lines", [
    ("", 1),                                                      # just falls off the end of the program
    ("argus_bench.stop()", 1),                                    # stops recording explicitly, then exits
    ("argus_bench.end()", 1),                                     # explicit end() with the declared function
    ("argus_bench.end(); argus_bench.stop()", 1),                 # both: still one
    ("import sys; sys.exit(3)", 1),                               # exits with a status code
    ("raise RuntimeError('agent crashed')", 1),                   # uncaught exception: exit hooks still run
])
def test_final_state_is_written_however_the_program_ends(tmp_path, ending, expected_end_lines):
    p = tmp_path / "r.jsonl"
    py(RECORD.format(path=str(p)) + ending + "\n")
    assert len(ends(p)) == expected_end_lines
    assert load_rollout(p)["end"]["env_state"] == {"final": "state"} and len(load_rollout(p)["calls"]) == 1


def test_hard_exit_skips_exit_hooks_but_the_calls_are_still_on_disk(tmp_path):
    p = tmp_path / "r.jsonl"
    py(RECORD.format(path=str(p)) + "import os; os._exit(0)\n")        # os._exit runs no hooks: this is a known limit, so call end() yourself before
    r = load_rollout(p)
    assert r["end"] is None and len(r["calls"]) == 1 and r["calls"][0]["env_state"] == {"final": "state"}


def test_killed_process_keeps_every_call_it_made(tmp_path):
    p = tmp_path / "r.jsonl"
    proc = subprocess.Popen([sys.executable, "-c", RECORD.format(path=str(p)) + "\nimport time; time.sleep(60)\n"])
    for _ in range(100):
        if p.exists() and p.read_text().count("\n") >= 2:
            break
        import time; time.sleep(0.1)
    proc.kill()
    proc.wait()
    assert len(load_rollout(p)["calls"]) == 1                          # calls are written as they happen, not at exit


def test_nothing_is_written_at_exit_if_the_program_made_no_calls(tmp_path):
    p = tmp_path / "r.jsonl"
    py(f"""
        import argus_bench
        argus_bench.record_to({str(p)!r}, library="x")
        argus_bench.env_state(lambda: {{"final": 1}})
    """)
    assert not p.exists()


def test_no_end_line_if_env_state_was_never_declared_and_end_never_called(tmp_path):
    p = tmp_path / "r.jsonl"
    py(f"""
        import argus_bench
        argus_bench.record_to({str(p)!r}, library="x")
        argus_bench.log([{{"role": "user", "content": "hi"}}], "ok", model="m")
    """)
    assert ends(p) == [] and load_rollout(p)["end"] is None


# ===================================================================== end() semantics
def test_several_end_lines_latest_wins_and_none_are_calls(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end({"v": 1})
    argus_bench.end({"v": 2})
    argus_bench.end({"v": 3})
    r = load_rollout(p)
    assert r["end"]["env_state"] == {"v": 3} and len(r["calls"]) == 1 and len(ends(p)) == 3


def test_end_before_any_call_is_allowed_and_check_says_there_are_no_calls(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.end({"early": True})
    r = load_rollout(p)
    assert r["calls"] == [] and r["end"]["env_state"] == {"early": True}
    assert check(p)[0] == ["no model calls in file"]


def test_end_after_stop_is_ignored_and_creates_nothing(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.stop()
    argus_bench.stop()                                                  # twice is fine
    before = p.read_bytes()
    argus_bench.end({"late": 1})
    argus_bench.log(MSGS, "late call", model="m")
    assert p.read_bytes() == before


@pytest.mark.parametrize("state", [
    {"a": 1}, {"nested": {"list": [1, 2.5, None, True, "x"], "deep": {"k": {"k": {"k": 1}}}}}, [1, 2, 3], "plain string", 42, 3.14, True,
    {"unicode": "héllo wörld — ✓ 你好 🙂", "newlines": "a\nb\r\nc\td"}, {"empty": {}, "list": []}, {"quote": 'he said "hi" \\ done'},
])
def test_end_state_round_trips_exactly(tmp_path, state):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end(state)
    assert load_rollout(p)["end"]["env_state"] == state


def test_end_state_that_is_not_json_is_stringified_not_fatal(tmp_path):
    class Odd:
        def __str__(self):
            return "odd"
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end({"o": Odd(), "s": {1, 2}, "p": tmp_path})
    got = load_rollout(p)["end"]["env_state"]
    assert got["o"] == "odd" and isinstance(got["s"], str) and str(tmp_path) in got["p"]


def test_a_huge_end_state_is_written_whole(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.log(MSGS, "ok", model="m")
    big = {"final_files": {f"f{i}.py": "x" * 100_000 for i in range(20)}}      # 2 MB
    argus_bench.end(big)
    assert load_rollout(p)["end"]["env_state"] == big


def test_end_with_none_and_a_declared_function_evaluates_the_function(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    box = {"n": 0}
    argus_bench.env_state(lambda: dict(box))
    argus_bench.log(MSGS, "ok", model="m")
    box["n"] = 9
    argus_bench.end()
    assert load_rollout(p)["end"]["env_state"] == {"n": 9}


def test_end_with_a_failing_function_records_the_error_and_does_not_raise(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.env_state(lambda: 1 / 0)
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end()
    assert "env_state error" in load_rollout(p)["end"]["env_state"]


def test_a_second_record_to_starts_a_fresh_file_with_its_own_end(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    start(a)
    argus_bench.env_state(lambda: {"who": "a"})
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end()
    start(b)                                                             # next rollout in the same process
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.end()
    argus_bench.stop()
    assert len(ends(a)) == 1 and len(ends(b)) == 1
    assert load_rollout(a)["end"]["env_state"] == {"who": "a"} and load_rollout(b)["end"]["env_state"] == {"who": "a"}
    assert len([o for o in lines(b) if "header" in o]) == 1


def test_end_written_by_another_process_than_the_one_that_made_the_calls(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.log(MSGS, "ok", model="m")
    argus_bench.stop()
    r = py(f"""
        import argus_bench
        argus_bench.record_to({str(p)!r}, library="harness")
        argus_bench.end({{"from": "the harness process"}})
    """)
    assert r.returncode == 0
    assert load_rollout(p)["end"]["env_state"] == {"from": "the harness process"} and len(load_rollout(p)["calls"]) == 1


# ===================================================================== env_state timing and lifecycle
def test_env_state_function_runs_exactly_once_per_call_for_every_call_style(tool_server, tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    count = {"n": 0}

    def fn():
        count["n"] += 1
        return {"n": count["n"]}
    argus_bench.env_state(fn)
    c = openai.OpenAI(base_url=tool_server, api_key="x")
    c.chat.completions.create(model="m", messages=[{"role": "user", "content": "FINISH"}])
    c.chat.completions.with_raw_response.create(model="m", messages=[{"role": "user", "content": "FINISH"}]).parse()
    argus_bench.log(MSGS, "manual", model="m")
    assert count["n"] == 3
    assert [x["env_state"] for x in load_rollout(p)["calls"]] == [{"n": 1}, {"n": 2}, {"n": 3}]


def test_env_state_is_taken_before_the_call_not_after(tmp_path, tool_server):
    """The server's reply cannot change what the state was when the call started."""
    p = tmp_path / "r.jsonl"
    start(p)
    box = {"phase": "before"}
    argus_bench.env_state(lambda: dict(box))
    c = openai.OpenAI(base_url=tool_server, api_key="x")
    c.chat.completions.create(model="m", messages=[{"role": "user", "content": "FINISH"}])
    box["phase"] = "after"
    c.chat.completions.create(model="m", messages=[{"role": "user", "content": "FINISH"}])
    assert [x["env_state"]["phase"] for x in load_rollout(p)["calls"]] == ["before", "after"]


def test_redeclaring_replaces_and_none_stops_attaching(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.env_state({"v": 1})
    argus_bench.log(MSGS, "a", model="m")
    argus_bench.env_state(lambda: {"v": 2})
    argus_bench.log(MSGS, "b", model="m")
    argus_bench.env_state(None)
    argus_bench.log(MSGS, "c", model="m")
    argus_bench.log(MSGS, "d", model="m", env_state={"v": "explicit"})      # an explicit value always wins
    argus_bench.env_state({"v": 4})
    argus_bench.log(MSGS, "e", model="m", env_state={"v": "explicit again"})
    assert [c.get("env_state") for c in load_rollout(p)["calls"]] == [{"v": 1}, {"v": 2}, None, {"v": "explicit"}, {"v": "explicit again"}]


def test_a_fixed_env_state_value_is_serialised_when_the_call_is_written(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    shared = {"files": []}
    argus_bench.env_state(shared)
    shared["files"].append("a.py")
    argus_bench.log(MSGS, "first", model="m")
    shared["files"].append("b.py")
    argus_bench.log(MSGS, "second", model="m")
    got = [c["env_state"]["files"] for c in load_rollout(p)["calls"]]
    assert got == [["a.py"], ["a.py", "b.py"]]


def test_threads_logging_concurrently_never_tear_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    argus_bench.env_state(lambda: {"t": threading.get_ident()})

    def work(i):
        for j in range(40):
            argus_bench.log([{"role": "user", "content": f"{i}-{j}" + "x" * 5000}], f"out {i}-{j}", model="m")
    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    argus_bench.end({"done": True})
    r = load_rollout(p)
    assert len(r["calls"]) == 320 and len({c["output"]["content"] for c in r["calls"]}) == 320 and r["end"]["env_state"] == {"done": True}


# ===================================================================== a tool loop that stops right after a tool call
def agent_loop(base_url, max_steps, final_result_store=None):
    """What a harness does: ask the model, run the tool it asks for, send the result back. Stops after max_steps."""
    c = openai.OpenAI(base_url=base_url, api_key="x")
    msgs = [{"role": "user", "content": "FINISH when you have run two commands"}]
    results = []
    for _ in range(max_steps):
        m = c.chat.completions.create(model="m", messages=msgs, tools=[{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}}]).choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            return msgs, results
        for tc in m.tool_calls:
            out = f"output of {json.loads(tc.function.arguments)['command']}"
            results.append(out)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return msgs, results                                                # step limit reached right after a tool call


def test_loop_that_finishes_has_every_tool_result_in_the_file(tool_server, tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    _, results = agent_loop(tool_server, max_steps=10)
    argus_bench.stop()
    r = load_rollout(p)
    assert len(r["calls"]) == 3 and r["calls"][-1]["output"]["content"] == "all done"
    tool_msgs = [m["content"] for m in r["calls"][-1]["messages"] if m["role"] == "tool"]
    assert tool_msgs == results == ["output of step 0", "output of step 1"]
    assert not any("tool call" in w for w in check(p)[1])


def test_loop_cut_off_by_a_step_limit_loses_the_last_result_unless_it_is_saved(tool_server, tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    msgs, results = agent_loop(tool_server, max_steps=2)                  # stops right after running the second tool
    argus_bench.stop()
    r = load_rollout(p)
    seen = [m["content"] for c in r["calls"] for m in c["messages"] if m["role"] == "tool"]
    assert results == ["output of step 0", "output of step 1"] and "output of step 1" not in seen     # the last result is in no record
    assert any("ends right after a tool call" in w for w in check(p)[1])                               # ...and check says so


def test_saving_the_last_result_with_end_keeps_it_and_silences_the_warning(tool_server, tmp_path):
    p = tmp_path / "r.jsonl"
    start(p)
    msgs, results = agent_loop(tool_server, max_steps=2)
    argus_bench.end({"last_tool_result": results[-1], "all_tool_results": results})
    argus_bench.stop()
    r = load_rollout(p)
    assert r["end"]["env_state"]["last_tool_result"] == "output of step 1"
    assert not any("tool call" in w for w in check(p)[1])
    assert "output of step 1" not in json.dumps(monitor_view(r))          # and the monitor still does not see it


def test_the_warning_is_part_of_the_submission_check(tool_server, tmp_path):
    sub = tmp_path / "sub"
    (sub / "rollouts").mkdir(parents=True)
    start(sub / "rollouts" / "a.jsonl")
    agent_loop(tool_server, max_steps=2)
    argus_bench.stop()
    (sub / "labels.json").write_text(json.dumps({"a.jsonl": False}))
    (sub / "verifier.py").write_text("from argus_bench import Verdict\ndef verify(rollout, artifacts_dir, llm):\n    return Verdict(False)\n")
    (sub / "verifier-requirements.txt").write_text("")
    problems, warnings, _ = check_submission(sub)
    assert not problems and any("ends right after a tool call" in w for w in warnings)


# ===================================================================== what verifiers and monitors see
def random_secret(rng):
    return "SECRET_" + "".join(rng.choice(string.ascii_letters) for _ in range(16))


def test_no_secret_in_env_state_end_annotations_or_header_ever_reaches_a_monitor(tmp_path):
    rng = random.Random(7)
    for i in range(25):
        p = tmp_path / f"r{i}.jsonl"
        secrets = [random_secret(rng) for _ in range(5)]
        argus_bench.record_to(p, rollout_id=secrets[0], library=secrets[1], **FULL)
        argus_bench.env_state(lambda s=secrets[2]: {"k": s, "deep": {"x": [s]}})
        for j in range(rng.randint(1, 4)):
            argus_bench.log([{"role": "user", "content": f"public task {j}"}], {"content": f"public answer {j}"}, model="m", sampling={"temperature": secrets[3]})
        argus_bench.end({"answer_key": secrets[4], "files": {"a.py": secrets[4]}})
        argus_bench.stop()
        argus_bench.env_state(None)
        annotate(p, {"label": secrets[1]}, call=0, by=secrets[0])
        annotate(p, secrets[3])
        seen = json.dumps(monitor_view(load_rollout(p)))
        assert "public task 0" in seen and "public answer 0" in seen
        for s in secrets:
            assert s not in seen, f"{s} leaked"


def test_verifier_sees_everything_the_monitor_does_not(tmp_path):
    p = tmp_path / "r.jsonl"
    argus_bench.record_to(p, rollout_id="rid", library="lib", **FULL)
    argus_bench.env_state(lambda: {"call_state": 1})
    argus_bench.log(MSGS, "answer", model="m", sampling={"temperature": 0.3})
    argus_bench.end({"end_state": 2})
    argus_bench.stop()
    annotate(p, {"note": 3}, call=0, by="alice")
    annotate(p, {"whole": 4})
    r = load_rollout(p)
    assert r["header"]["rollout_id"] == "rid" and r["calls"][0]["env_state"] == {"call_state": 1} and r["calls"][0]["sampling"] == {"temperature": 0.3}
    assert r["end"]["env_state"] == {"end_state": 2} and r["calls"][0]["annotations"][0]["data"] == {"note": 3} and r["annotations"][0]["data"] == {"whole": 4}
    assert monitor_view(r) == {"calls": [{"messages": MSGS, "tools": None, "output": {"content": "answer"}}]}


def test_a_verifier_that_needs_the_end_state_fails_loudly_on_a_rollout_without_one(tmp_path):
    sub = tmp_path / "sub"
    (sub / "rollouts").mkdir(parents=True)
    for name, with_end in (("a.jsonl", True), ("b.jsonl", False)):
        start(sub / "rollouts" / name)
        argus_bench.log(MSGS, "ok", model="m", sampling={"temperature": 1})
        if with_end:
            argus_bench.end({"flag": True})
        argus_bench.stop()
    (sub / "labels.json").write_text(json.dumps({"a.jsonl": True, "b.jsonl": False}))
    (sub / "verifier.py").write_text("from argus_bench import Verdict\ndef verify(rollout, artifacts_dir, llm):\n    return Verdict(rollout['end']['env_state']['flag'])\n")
    (sub / "verifier-requirements.txt").write_text("")
    problems, _, report = check_submission(sub, run=True, log=lambda *_: None)
    errors = report["verifier_run"]["errors"]
    assert not problems and len(errors) == 1 and errors[0].startswith("b.jsonl") and report["verifier_run"]["rollouts_run"] == 1


def test_one_verifier_using_call_state_end_state_and_annotations_together(tmp_path):
    sub = tmp_path / "sub"
    (sub / "rollouts").mkdir(parents=True)
    plan = {"bad.jsonl": (True, True), "ok.jsonl": (False, False), "flagged_by_human.jsonl": (False, True)}     # (changed files, annotator says misbehavior)
    labels = {}
    for name, (changed, human) in plan.items():
        start(sub / "rollouts" / name)
        argus_bench.env_state(lambda c=changed: {"changed": c})
        argus_bench.log(MSGS, "step", model="m", sampling={"temperature": 1})
        argus_bench.log(MSGS + [{"role": "assistant", "content": "step"}], "done", model="m", sampling={"temperature": 1})
        argus_bench.end({"files_changed": changed})
        argus_bench.stop()
        argus_bench.env_state(None)
        if human:
            annotate(sub / "rollouts" / name, {"misbehavior": True}, call=1, by="annotator")
        labels[name] = changed or human
    (sub / "labels.json").write_text(json.dumps(labels))
    (sub / "verifier.py").write_text(textwrap.dedent("""
        from argus_bench import Verdict, Span
        def verify(rollout, artifacts_dir, llm):
            from_state = rollout["end"]["env_state"]["files_changed"] or any(c["env_state"]["changed"] for c in rollout["calls"])
            from_human = any(a["data"].get("misbehavior") for c in rollout["calls"] for a in c["annotations"])
            return Verdict(from_state or from_human, "state" if from_state else ("human" if from_human else ""), [Span(1)] if from_human else [])
    """))
    (sub / "verifier-requirements.txt").write_text("")
    problems, _, report = check_submission(sub, run=True, log=lambda *_: None)
    assert not problems and report["verifier_run"]["accuracy"] == 1.0 and report["verifier_run"]["true_positive"] == 2
