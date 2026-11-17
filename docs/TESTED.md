# What has been tested

Tested means: run for real against the library or server, and the resulting file passed `argus-bench check` and contained the expected content.

| way to run the model | route | status |
|---|---|---|
| vLLM 0.11 server (sync, stream, async, tool calls, compaction) | A (`record_to`) | tested |
| llama-cpp-python server | A | tested |
| Ollama 0.40 server (tool call with reasoning, streaming) | A | tested |
| LiteLLM in front of a server (sync, stream) | A | tested (async not tried) |
| agent in a separate process, `ARGUS_BENCH_LOG` environment variable | B | tested with a mock server in a clean install of the wheel; not yet in a real container |
| transformers in-process | C (`log`) | tested |
| vLLM offline `LLM`, llama-cpp-python in-process | C | tested earlier; the snippets in USAGE.md were retyped and not re-run |
| SGLang 0.5.5 server (tool call with reasoning split, stream, async, compaction, `env_state`, and an agent in a subprocess via `ARGUS_BENCH_LOG`) | A, B | tested (openai 2.6.1 on the client side) |
| SGLang `Engine` in-process | C | not yet tested |
| Responses API, non-Python clients, SDKs other than `openai` | | not supported (write the file yourself, see FORMAT.md) |

`openai` package versions: the capture tests pass on 1.6.1, 1.12.0, 1.30.5, 1.54.5, 1.82.0, 1.109.1, 2.0.0, 2.30.0 and 3.49.0 (spot checks, not every
release). 1.0.0 cannot be used (its `create` has no `tools` argument). Only Python 3.12 has been run.
