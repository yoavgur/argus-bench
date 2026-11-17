# argus-bench

Record every model call of an agent run, whatever inference library runs the model, in one plain JSON Lines file.
For each call it stores what the model was shown and what it produced.

```
pip install argus-bench
```

```python
import argus_bench
argus_bench.record_to("rollout.jsonl", revision="<full model commit hash>", library="vllm 0.23.0 (server)",
                      dtype="bfloat16", quantization="none", chat_template="default")
# ... run your agent as usual: every OpenAI-SDK chat call (vLLM, SGLang, llama.cpp, Ollama, LiteLLM, LangChain; sync, async, streaming) is recorded
```

Model runs inside your own process (transformers, vLLM `LLM`, SGLang `Engine`, llama-cpp-python)? One call after each generation:

```python
argus_bench.log(messages, generated_text, tools=tools, model="Qwen/Qwen3-1.7B", sampling={"temperature": 0.7, "top_p": 0.95})
```

Agent you cannot edit, or one that runs in a container? Set `ARGUS_BENCH_LOG=/path/rollout.jsonl` in its environment.
Check the result with `argus-bench check rollout.jsonl`.

* [docs/USAGE.md](docs/USAGE.md): how to record with every supported way of running a model
* [docs/EXAMPLE.md](docs/EXAMPLE.md): a complete worked example (record, verifier, submission, annotations)
* [docs/SUBMISSION.md](docs/SUBMISSION.md): verifier, `Verdict`, labels, evidence, and `argus-bench check-submission`
* [docs/FORMAT.md](docs/FORMAT.md): the exact file format, so you can produce it without this package
* [docs/TESTED.md](docs/TESTED.md): what has and has not been tested
