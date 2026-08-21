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
uv run python -m unittest discover -s tests -t .          # 142 tests, offline, ~3s
AGENTFIX_LLM_TESTS=1 uv run python -m unittest discover -s tests -t .   # + live-model tests
```

The whole suite runs with no model process anywhere: `llm/fake.py` is a real `BaseChatModel`
that returns a scripted list of replies, so the tests drive the **real** graph against the
**real** tools in a real temp directory. Only the model is replaced.

## Reading order

1. `tools/base.py` — what a tool is, and the limits on what it may return
2. `tasks/loader.py` — what a task is; the copy-to-tempdir context manager
3. `tools/fs.py` — `list_files`, `read_file`, `write_file`
4. `tools/tests_tool.py` — `run_tests`, the agent's only oracle
5. `agent/graph.py` — **the agent.** If you read one file, read this one.
6. `runner.py` — how the pieces are wired together

## What the framework gave us

- `ToolNode` replaces the hand-written `dispatch`, including its unknown-tool and
  bad-argument observations, and answering several calls in one turn.
- `add_messages` makes the history append-only by construction.
- State, checkpointing and streaming, for free.
- `ChatOpenAI` parses tool calls, token usage and malformed arguments.

## What it did not

- **`handle_tool_errors` defaults to letting a tool's exception kill the run.** The original
  guaranteed `dispatch` never raises. You have to opt back in — and passing a *string* rather
  than `True` silently discards the specific error, so the model stops being told which
  argument it forgot.
- **`invalid_tool_calls` are ignored entirely.** A tool call whose JSON did not parse gets no
  reply message at all, and the API requires an answer to every call. A 12B model gets that
  JSON wrong often enough for this to matter.
- **`max_tokens` does not reach Ollama.** `ChatOpenAI` aliases it to OpenAI's newer
  `max_completion_tokens`, which Ollama's `/v1` ignores. Measured, asking for 8 tokens:
  `max_completion_tokens=8` → 692 generated; `max_tokens=8` → 8 generated. Passing it the
  framework's way silently removes the cap on a single reply.
- **The loop guard.** No framework knows that a repeated identical call means your model is
  stuck.
- **The step budget.** `recursion_limit` counts node executions, not model turns.

## Parity

The port is meant to change the code and not the agent. Same task, same model:

| | no framework | LangGraph |
|---|---|---|
| verdict | SOLVED | SOLVED |
| steps | 8 | 8 |
| tokens | 8,626 | 8,684 |
| peak context | 1,388 | 1,383 |
| turns with reasoning | 0 of 7 | 0 of 7 |

Same tool sequence, step for step. The small token difference is the extra
`tests/__init__.py` line in `list_files` output and unittest's shorter failure report.

**The agent does not reason.** Seven tool-calling turns, seven `(NO REASONING)` markers, and
the only prose arrives at step 8 *after* the fix is already verified. It also reads all three
source files despite being told not to, and finds the bug at step 4 but keeps looking. That is
the Act-only baseline from the ReAct paper, and closing that gap is the next act.
