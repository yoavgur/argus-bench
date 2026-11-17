# Writing a submission: verifier, labels, evidence

A submission is a folder (zipped) with:

```
rollouts/            one .jsonl file per rollout (the format in FORMAT.md), misbehavior and benign rollouts of the same task
labels.json          {"rollout_01.jsonl": true, "rollout_02.jsonl": false, ...}     true = contains the misbehavior
verifier.py          def verify(rollout: dict, artifacts_dir: Path, llm: OpenAI) -> Verdict
verifier-requirements.txt
verifier_config.json only if verify() uses `llm`: {"model": "<hf model id>", "revision": "<full 40-char commit hash>"}
artifacts/           anything else the verifier needs
```

Use one file per rollout. If the agent runs through the environment-variable route, give each rollout its own `ARGUS_BENCH_LOG` path.

## The verifier

```python
from pathlib import Path
from argus_bench import Verdict, Span

def verify(rollout: dict, artifacts_dir: Path, llm) -> Verdict:
    ...
    return Verdict(misbehavior=True,
                   explanation="The model edited the tests so they pass instead of fixing the bug.",
                   spans=[Span(call=7, location="output.tool_calls.0.arguments", start=12, end=60)])
```

`Verdict(misbehavior: bool, explanation: str = "", spans: list[Span] = [])`. `explanation` and `spans` are optional but encouraged.

`Span(call, location="output.content", start=None, end=None, text=None)` points into the transcript:
* `call`: index into `rollout["calls"]` (0-based)
* `location`: which text inside that call: `output.content`, `output.reasoning_content`, `output.tool_calls.<i>.arguments`, or `messages.<i>.content`
* `start`, `end`: optional character range inside that text (end exclusive). Leave both out to point at the whole text.
* `text`: optional copy of the quoted text

`llm` is a standard OpenAI-compatible client. Use it for any LLM call and name the model you rely on in `verifier_config.json` (open-weight models only).

## What the verifier receives

`rollout` is what `argus_bench.load_rollout(path)` returns, and it never contains the label:

```python
{"id": "rollout_01",                         # file name without extension; use it to find artifacts/rollout_01/
 "header": {"rollout_id": ..., "library": ..., "models": ...},
 "calls": [{"model": ..., "messages": [...], "tools": [...], "sampling": {...}, "output": {...},
            "finish_reason": ..., "env_state": ...}, ...],
 "end": {"t": ..., "env_state": ...}}        # or None
```

## What monitors see

Monitors see the transcript only: `argus_bench.monitor_view(rollout)` returns `{"calls": [{"messages", "tools", "output", "finish_reason"}, ...]}`.
Hidden from monitors: `env_state`, the end state, annotations, the header (rollout_id, library, revision, ...), sampling settings, timestamps and the rollout id.
So anything you put into `env_state`, `end()` or `artifacts/` is evidence for your verifier only.

## Evidence the monitor cannot see

A complete walkthrough is in [EXAMPLE.md](EXAMPLE.md). Four places, use whichever fits:

1. **`argus_bench.env_state(...)`**: state at the start of each model call (a function or a value; see USAGE.md). Ends up in `rollout["calls"][i]["env_state"]`.
2. **`argus_bench.end(state)`**: state at the end of the run, once per rollout: final contents of edited files, the log of commands the model ran, an answer key,
   labels from human annotators. Ends up in `rollout["end"]["env_state"]`. If you declared `env_state(...)` and never call `end()`, the final state is written
   automatically when the program exits.
   ```python
   argus_bench.end({"final_files": {"solution.py": open("solution.py").read()}, "answer_key": "42"})
   ```
3. **`artifacts/`**: bigger files. Suggested layout: `artifacts/<rollout id>/...`, and the verifier reads `artifacts_dir / rollout["id"] / "..."`.
   Record paths or hashes in `env_state`/`end` if you want a pointer from the rollout.

4. **`argus_bench.annotate(...)`**: added later, by a person or another program (see the next section).

Note: a tool result appears in the file only as an input message of the next model call. If a run can end right after a tool call, put that last result in `end(...)`.

Keep `env_state`/`end` values small and JSON-able.

## Adding evidence after the run (human annotators, second graders)

Evidence can also be added after the fact, to the rollout as a whole, to one model call, or to one input message of a call. The original lines of the file are never
changed: each annotation is a new line appended at the end.

```python
argus_bench.annotate("rollouts/rollout_07.jsonl", {"misbehavior": True, "note": "deletes the backup"}, call=3, by="annotator_2")             # about call 3
argus_bench.annotate("rollouts/rollout_07.jsonl", {"misleading": "the user never asked for this"}, call=3, message=5, by="annotator_2")    # about message 5 of call 3
argus_bench.annotate("rollouts/rollout_07.jsonl", {"overall": "clear sabotage"}, by="annotator_3")                                          # whole rollout
```
```
argus-bench annotate rollouts/rollout_07.jsonl --call 3 --by annotator_2 --data '{"misbehavior": true}'      # same from the command line
```

`data` is any JSON value. `annotate` refuses a call or message index that does not exist, and `argus-bench check` flags annotations that point nowhere (for example after
hand edits). The verifier receives them as `rollout["calls"][i]["annotations"]` (a list of `{"message", "data", "by", "t"}`, empty if none) and `rollout["annotations"]`
for the whole rollout. Monitors never see them.

## Check your submission before uploading

```
argus-bench check-submission my_submission/            # or my_submission.zip
argus-bench check-submission my_submission/ --run      # also runs verifier.py on every rollout and compares with labels.json
argus-bench check-submission my_submission/ --run --llm-base-url http://localhost:8000/v1     # if the verifier uses an LLM
```

It checks the layout, that `labels.json` covers exactly the rollout files, that both labels are present, that every rollout file is well-formed, that `verifier.py`
defines `verify(rollout, artifacts_dir, llm)`, that the requirements and (if `llm` is used) config files exist, and that the config names a model and a full revision hash.
With `--run` it executes your verifier on each rollout and prints, per rollout, the verdict against the label, plus accuracy and how many verdicts lack an explanation.
`--run` executes the submitted code: only run it on code you trust.
