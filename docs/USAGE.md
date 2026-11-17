# Using argus-bench

```
pip install argus-bench      # pure Python, no dependencies
```

Pick the row that matches how your agent calls the model. All of them write the same file ([FORMAT.md](FORMAT.md)).

| how the model is run | what you do |
|---|---|
| a server with an OpenAI-compatible API (vLLM, SGLang, llama.cpp / llama-cpp-python server, Ollama, LM Studio, TGI, LiteLLM proxy, a hosted API) and the agent uses the `openai` Python package (directly, or through LiteLLM, LangChain, ...) | [A. one line, `record_to`](#a-openai-compatible-server) |
| same, but the agent is a program you cannot edit, or runs in another process or container | [B. environment variable](#b-agent-you-cannot-edit-subprocess-container) |
| the model runs inside your own Python process (transformers, vLLM `LLM`, SGLang `Engine`, llama-cpp-python, your own loop) | [C. `log()` after each generation](#c-model-runs-in-your-process) |
| anything else (another language, raw HTTP, another SDK) | [D. write the file yourself](#d-anything-else) |

Always declare what you know about the run: `rollout_id`, `library`, and for the model `revision` (full commit hash), `dtype`, `quantization`, `chat_template`.
`argus-bench check` warns about missing ones.

## A. OpenAI-compatible server

Add this before the agent starts, in the process that makes the model calls:

```python
import argus_bench
argus_bench.record_to("rollout.jsonl",
                      rollout_id="task-17-run-3",
                      revision="<full 40-char hf commit hash of the weights>",
                      library="vllm 0.23.0 (server)", dtype="bfloat16", quantization="none", chat_template="default")
```

Everything the `openai` package sends to `/chat/completions` from then on is recorded: `client.chat.completions.create`, `with_raw_response`,
sync, async and streaming, and libraries built on the SDK (LiteLLM, LangChain, Agents SDK). Your code gets the normal responses back, unchanged.
Nothing else about how you start or call the server changes.

If the run uses more than one model, declare each one by name instead of the single `revision=`/`dtype=`/... keys:

```python
argus_bench.record_to("rollout.jsonl", rollout_id="task-17-run-3", library="vllm 0.23.0 (server)",
                      models={"Qwen/Qwen3-1.7B": {"revision": "<hash>", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"},
                              "Qwen/Qwen3-0.6B": {"revision": "<hash>", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}})
```

Notes:
* The model server does not need any flag or plugin.
* It hooks one internal function of the `openai` package. Capture tests pass on openai 1.6.1 up to 3.x. If a future version breaks the hook,
  `record_to` prints a warning, and at program exit another warning appears if no call was recorded.
* Only `/chat/completions` is recorded. The Responses API is not.
* The server must return reasoning in `reasoning_content` (vLLM: `--reasoning-parser`, SGLang: `--reasoning-parser`) to have it recorded separately.
  Otherwise it stays inside `content`, which is fine.

## B. Agent you cannot edit (subprocess, container)

Install `argus-bench` in the Python environment the agent runs in and set environment variables for its process:

```
ARGUS_BENCH_LOG=/path/rollout.jsonl
ARGUS_BENCH_META='{"revision": "<hash>", "library": "vllm 0.23.0 (server)", "dtype": "bfloat16", "quantization": "none", "chat_template": "default"}'
```

Every Python process started with those variables records its `openai` calls to the file (the package ships a `.pth` file that starts the
recording at interpreter start-up). Several processes can write to the same file. Processes that make no model call write nothing.
This is the same as route A, so the same notes apply.

## C. Model runs in your process

Call `argus_bench.log(messages, output, ...)` after each generation. `messages` is the full chat input the model got; `output` is the generated
text, or a dict `{"content", "reasoning_content", "tool_calls"}`.

```python
argus_bench.record_to("rollout.jsonl", revision="<hash>", library="transformers 5.22 (in-process)", dtype="bfloat16",
                      quantization="none", chat_template="default", rollout_id="task-17-run-3")
```

**transformers**
```python
ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
gen = model.generate(**ids, do_sample=True, temperature=0.7, top_p=0.95, top_k=20, max_new_tokens=300)
text = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
argus_bench.log(messages, text, tools=tools, model="Qwen/Qwen3-1.7B",
                sampling={"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 300})
```

**vLLM offline (`LLM`)**
```python
sp = SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=300, seed=7)
out = llm.chat(messages, sp, tools=tools)[0].outputs[0]
argus_bench.log(messages, out.text, tools=tools, model="Qwen/Qwen3-1.7B",
                sampling={"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 300, "seed": 7},
                finish_reason=out.finish_reason)
```

**SGLang `Engine`**
```python
ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
o = engine.generate(input_ids=list(ids), sampling_params={"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_new_tokens": 300})
argus_bench.log(messages, o["text"], tools=tools, model="Qwen/Qwen3-1.7B",
                sampling={"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 300})
```

**llama-cpp-python (`Llama`)**
```python
r = llm.create_chat_completion(messages=messages, tools=tools, temperature=0.7, top_p=0.95, max_tokens=300)
m = r["choices"][0]["message"]
argus_bench.log(messages, {"content": m.get("content"), "tool_calls": [{"id": t["id"], "name": t["function"]["name"], "arguments": t["function"]["arguments"]} for t in m.get("tool_calls", [])]},
                tools=tools, model="Qwen/Qwen3-1.7B", sampling={"temperature": 0.7, "top_p": 0.95, "max_tokens": 300}, finish_reason=r["choices"][0]["finish_reason"])
```

If your code does not parse tool calls or reasoning, pass the raw generated text as `output`; do not strip or reformat it.
Extra keyword arguments to `log` (such as `finish_reason=`) are written into the record.

## D. Anything else

Write the JSON Lines file yourself. [FORMAT.md](FORMAT.md) is the full specification and has a minimal example.

## Environment state (optional)

If you want to record what the environment looked like at each call, declare it once:

```python
argus_bench.env_state(lambda: {"files": sorted(os.listdir(workdir)), "git_head": git_head()})   # function, called at the start of every model call
argus_bench.env_state({"task": "fix-bug-12"})                                                     # or a fixed value
```

The value ends up as `env_state` on every record. It is evaluated in the process that called `record_to`, so that process must be able to see the
environment (read the files, or ask the sandbox). With route B (environment variables) there is no code to declare it in, so use a function that
reads a state file your harness keeps up to date, or pass `env_state=` to `log`.

## Check your file

```
argus-bench check rollout.jsonl
rollout.jsonl: 4 calls  {"models": ["Qwen/Qwen3-1.7B"], "calls_extending_previous": 2, "calls_with_new_context (compaction / new thread)": 1}
  OK
```

It reports problems (no messages, empty output), warnings (missing header declarations, tool-call arguments that are not valid JSON,
missing sampling settings) and how many calls continue the previous call's input vs. start a new context (compaction, a new conversation, or a
parallel thread). No model and no network needed. Exit code 1 if there are problems.

## What was tested

See [TESTED.md](TESTED.md).
