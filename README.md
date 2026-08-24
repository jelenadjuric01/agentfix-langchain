# agentfix (LangGraph edition)

The same teaching coding agent as the no-framework original, rebuilt on LangGraph. It fixes
real bugs, locally, for $0, using [JetBrains
Mellum2](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF) served by Ollama.

This repo exists to be **compared** against the no-framework one. The interesting content is
not "how to use LangGraph" — it is which parts of a hand-written agent a framework absorbs,
which parts it leaves you holding, and which parts it quietly breaks if you are not watching.

## Setup

```bash
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
ollama create agentfix-mellum2 -f Modelfile
uv sync --extra dev
uv run agentfix doctor

# optional: the create_agent side-by-side in agent/prebuilt.py and its tests
uv sync --extra dev --extra prebuilt
```

## Use

```bash
uv run agentfix solve tasks/workshop/01-shopcart --verbose
uv run agentfix eval --suite workshop
uv run agentfix eval --suite humanevalfix --limit 5
```

## Tests

unittest only, no pytest anywhere — including inside the task fixtures the agent fixes.

```bash
uv run python -m unittest discover -s tests -t .          # 194 tests, offline, ~5s
AGENTFIX_LLM_TESTS=1 uv run python -m unittest discover -s tests -t .   # + live-model tests
```

The whole suite runs with no model process anywhere: `llm/fake.py` is a real `BaseChatModel`
that returns a scripted list of replies, so the tests drive the **real** graph against the
**real** tools in a real temp directory. Only the model is replaced.

## Reading order

1. `tools/base.py` — what a tool is, the limits on what it may return, and the artifact
   channel it reports through
2. `tasks/loader.py` — what a task is; the copy-to-tempdir context manager
3. `tools/fs.py` — `list_files`, `read_file`, `write_file`
4. `tools/tests_tool.py` — `run_tests`, the agent's only oracle
5. `agent/state.py` — what the graph carries between nodes, and the reducers that combine it
6. `agent/graph.py` — **the agent.** If you read one file, read this one.
7. `runner.py` — how the pieces are wired together

Then, if you want the argument rather than the code: `agent/prebuilt.py` builds the same agent
out of `langchain.agents.create_agent` and its middleware, and documents which of the three
invariants that buys you. Needs `--extra prebuilt`; `tests/test_prebuilt.py` pins both halves,
including a test that fails if the framework ever closes the `invalid_tool_calls` gap.

## What the framework gave us

- `ToolNode` replaces the hand-written `dispatch`, including its unknown-tool and
  bad-argument observations, and answering several calls in one turn. It gets the whole
  surviving batch per turn; the only thing wrapped around it is the loop guard.
- `add_messages` makes the history append-only by construction.
- Reducers on the rest of the state — `operator.add` for the token counters, a two-argument
  `keep_larger` for the peak — so a node returns a delta and never reads the old value.
- Callbacks carry the trace. `agent/trace.py` is a `BaseCallbackHandler` handed to the graph
  once, so the nodes contain no tracing code at all.
- Checkpointing: `InMemorySaver` snapshots the state after every node, which makes a run
  resumable and its history inspectable step by step.
- `ChatOllama` parses tool calls, token usage and malformed arguments.

## What it did not

- **`handle_tool_errors` defaults to letting a tool's exception kill the run.** The original
  guaranteed `dispatch` never raises. You have to opt back in — and passing a *string* rather
  than `True` silently discards the specific error, so the model stops being told which
  argument it forgot.
- **`invalid_tool_calls` are ignored entirely.** A tool call whose JSON did not parse gets no
  reply message at all, and the API requires an answer to every call. A 12B model gets that
  JSON wrong often enough for this to matter.
- **The loop guard.** No framework knows that a repeated identical call means your model is
  stuck.
- **The step budget — on LangGraph.** `recursion_limit` counts node executions, not model
  turns. On LangChain 1.x this one has *moved*: `ModelCallLimitMiddleware(run_limit=N)` counts
  exactly what `AgentState.step` counts. See `agent/prebuilt.py`, including the measurement
  showing it is silently ignored if you order the middleware wrong.
- **Checkpointing is only as good as what you put in the state.** The test verdict used to
  live on the `run_tests` tool object. The graph was resumable; the agent was not — a resumed
  run rebuilt that tool empty and reported a solved task unsolved. The verdict now travels as
  a `ToolMessage` artifact into `AgentState.tests_passed`, and the fix deleted the
  `on_write=run_tests.invalidate` wiring along with it.
- **The wrong integration will lie to you.** This repo used `ChatOpenAI` against Ollama's
  `/v1` endpoint, because that kept the wire format byte-identical to the no-framework
  original. Two of the three settings that decide whether the agent works were being
  discarded in transit, silently. Measured, same server:

  | | `ChatOpenAI` via `/v1` | `ChatOllama` |
  |---|---|---|
  | cap on one reply | `max_completion_tokens=8` → 692 tokens | `num_predict=8` → 8 tokens |
  | context window | `options` dropped; `ollama ps` says 4096 | `num_ctx=8192` → `ollama ps` says 8192 |

  A compatibility endpoint accepts the requests it does not honour. The Modelfile still
  works, but it is belt and braces now rather than the only way to set a context window.

## Parity

The port is meant to change the code and not the agent. Same task, same model:

| | no framework | LangGraph |
|---|---|---|
| verdict | SOLVED | SOLVED |
| steps | 8 | 8 |
| tokens | 8,626 | 8,646 |
| peak context | 1,388 | 1,384 |
| turns with reasoning | 0 of 7 | 0 of 7 |

Same tool sequence, step for step. The small token difference is the extra
`tests/__init__.py` line in `list_files` output and unittest's shorter failure report.

Worth knowing that this column was re-measured after the agent was rebuilt to use the
framework properly — state reducers, callback tracing, one batched `ToolNode` call per turn, a
checkpointer, and Ollama's native client instead of the OpenAI-compatible one. Eight steps,
8,646 tokens, and the same tool sequence, from a graph whose nodes no longer contain any
tracing, accumulation or dispatch code. `pass@1 = 1.00` on the three workshop tasks.

The wire format is no longer byte-identical to the original, and that is the deliberate trade:
`ChatOpenAI` matched the original's bytes and silently dropped the reply cap and the context
window on the way out. See **What it did not**, above.

**The agent does not reason.** Seven tool-calling turns, seven `(NO REASONING)` markers, and
the only prose arrives at step 8 *after* the fix is already verified. It also reads all three
source files despite being told not to, and finds the bug at step 4 but keeps looking. That is
the Act-only baseline from the ReAct paper, and closing that gap is the next act.
