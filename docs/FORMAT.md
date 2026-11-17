# The rollout file format

Everything `argus-bench` writes is one plain text file in **JSON Lines** format: UTF-8, one JSON object per line, append-only.
You do not need the package to produce it. Any program that writes this format is a valid submission.

The file holds one **rollout** (one agent run). It consists of header lines followed by one **record per model call**.

## Header line

```json
{"header": {"argus_bench": "0.1.0", "python": "3.12.7", "started": 1794397938.55,
            "rollout_id": "task-17-run-3", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "library": "vllm 0.23.0 (server)", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}}
```

A line is a header line if its only key is `header`. Several header lines may exist (one per process that wrote to the file); readers merge them.
Put every fact about the *run* here, not in the records.

| key | meaning | required |
|---|---|---|
| `rollout_id` | your identifier for this rollout | recommended |
| `revision` | the full 40-character Hugging Face commit hash of the weights (not a branch, not a short hash) | yes, if the model is on the Hub |
| `library` | inference library and version, and how it was run (`vllm 0.23.0 (server)`, `transformers 5.22 (in-process)`, ...) | yes |
| `dtype` | numeric precision the model ran in (`bfloat16`, `float16`, `float32`) | yes |
| `quantization` | `"none"`, or what was used (`"gguf-q4_k_m"`, `"awq"`, `"fp8"`, ...) | yes |
| `chat_template` | `"default"` if the library used the model's own chat template, else a name or hash of the template you used | yes |
| `argus_bench`, `python`, `started`, `openai_sdk` | written automatically by the package; leave out if you write the file yourself | no |

Any additional keys are allowed.

**Several models in one run.** If the agent calls more than one model (for example a second, smaller model that summarises the context), declare each
model by the name used in the records' `model` field instead of the four single-model keys above:

```json
{"header": {"rollout_id": "task-17-run-3", "library": "vllm 0.23.0 (server)",
            "models": {"Qwen/Qwen3-1.7B": {"revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"},
                       "Qwen/Qwen3-0.6B": {"revision": "c1899de289a04d12100db370d81485cdf75e47ca", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}}}}
```

Every model that appears in a record must be declared. For a run with one model you can use either form.

## Record line (one per model call)

```json
{"t": 1794397941.88,
 "model": "Qwen/Qwen3-1.7B",
 "messages": [{"role": "system", "content": "You are a shell agent."},
              {"role": "user", "content": "List the files and tell me how many there are."}],
 "tools": [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
 "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 300, "seed": 7},
 "output": {"reasoning_content": "The user wants...", "content": "",
            "tool_calls": [{"id": "call_1", "name": "bash", "arguments": "{\"command\": \"ls\"}"}]},
 "finish_reason": "tool_calls",
 "env_state": {"step": 1}}
```

| field | type | meaning |
|---|---|---|
| `t` | number | unix time (seconds) of the call |
| `model` | string | the model name that was asked for (`Qwen/Qwen3-1.7B`) |
| `messages` | list | **the complete input the model was shown for this call**, OpenAI chat format (see below). Never a diff against the previous call. |
| `tools` | list or null | the tool definitions the model was given, OpenAI function-tool format |
| `sampling` | object | every sampling setting that was in effect and that you know: `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `stop`. Leave out what you did not set. |
| `output` | object | what the model produced, see below. Must contain at least one of its keys. |
| `finish_reason` | string | `stop`, `length` (cut off by the token limit), `tool_calls`, ... as your library reports it. Recommended; it matters for `length`. |
| `env_state` | any JSON | optional: state of the environment at the start of this call (see below) |
| `stream` | boolean | optional: `true` if the call was streamed (the package sets it) |
| `source` | string | optional: set by the package (`openai-sdk`) |

### `messages`

OpenAI chat format, in the order the model saw them, including everything the agent added since the last call:

```json
[{"role": "system", "content": "..."},
 {"role": "user", "content": "..."},
 {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
 {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"}]
```

The assistant turns inside `messages` are the model's earlier outputs as the agent resent them. Write them the way they were sent to the model,
including any rewriting the agent did (e.g. reasoning removed, content trimmed).

### `output`

What the model produced in *this* call, as separate fields. Leave out empty ones.

| key | meaning |
|---|---|
| `content` | the visible text. If your server does not split reasoning from text, put everything the model wrote here, exactly as generated (including any `<think>...</think>`). |
| `reasoning_content` | the reasoning text, only if your server returned it separately |
| `tool_calls` | list of `{"id": ..., "name": ..., "arguments": ...}`. `arguments` is the **string** the model produced, kept as-is even if it is not valid JSON. If your library does not parse tool calls, leave them inside `content` as raw text. |

Write the model's text exactly as it came out. Do not trim, normalise, pretty-print or re-serialise it.

## End line (optional)

A rollout may end with one line holding the state of the environment when the run finished:

```json
{"end": {"t": 1794397999.1, "env_state": {"final_files": {"solution.py": "..."}, "answer_key": "42"}}}
```

If several end lines exist (several processes), the latest one counts. It is not a model call. See SUBMISSION.md for what it is for.

## Annotation lines (optional)

Information added after the run, never by editing existing lines:

```json
{"annotation": {"call": 3, "message": 5, "data": {"misbehavior": true, "note": "deletes the backup"}, "by": "annotator_2", "t": 1794398500.2}}
```

`call` (0-based index of a model call; `null` = the whole rollout), `message` (0-based index into that call's `messages`; needs `call`), `data` (any JSON), `by`, `t`.
Not model calls. See SUBMISSION.md.

## Rules

1. One record per model call, in call order. If a call returns several choices (`n > 1`), write one record per choice.
2. A streamed call is one record: the final assembled message.
3. Every record is self-contained (`messages` is the whole input). **Context compaction needs no special handling**: after the agent compacts,
   the next record simply has a shorter or different `messages`. There is no compaction marker.
4. Calls that failed with an error (no output) are not recorded.
   A tool result is recorded only as an input message of the *next* model call. If the run stops right after a tool call (step limit, timeout, crash),
   that last result is in no record: save it with `argus_bench.end({...})` if your verifier needs it.
5. Subagents, parallel branches and several processes may write into the same file; each record stands on its own. Write each line with a single
   write call so lines from different processes do not interleave.
6. `env_state` is whatever you declare it to be (a dict of facts, a hash, a file path). It is opaque, self-reported and should stay small;
   keep large state in a separate file and record the path or hash.

## Minimal valid file

```
{"header": {"library": "my-engine 1.0", "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}}
{"model": "Qwen/Qwen3-1.7B", "messages": [{"role": "user", "content": "Hi"}], "sampling": {"temperature": 0.7}, "output": {"content": "Hello!"}, "finish_reason": "stop"}
```

Check a file with `argus-bench check rollout.jsonl` (needs the package; no model, no network).
