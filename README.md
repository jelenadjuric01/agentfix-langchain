# agentfix (LangGraph edition)

A teaching repository for a workshop that shows developers new to agents how a coding agent
actually works, by having them build the two decisions that matter. You write `route_after_agent`
— where a run is allowed to end, and on whose word — and the loop guard that catches a model
which has stopped learning. Then you watch it fix real bugs, locally, for $0, using [JetBrains
Mellum2](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF) served by Ollama.

This is the **framework** edition, built on LangGraph. A sibling repository builds the same agent
with no framework at all, and the interesting content is not "how to use LangGraph" — it is which
parts of a hand-written agent a framework absorbs, which parts it leaves you holding, and which
parts it quietly breaks if you are not watching. See [What the framework gave
us](#what-the-framework-gave-us).

> **You are on `main`, the exercise branch.** Two functions in
> `src/agentfix/agent/graph.py` are deliberately unwritten, so the exercise tests fail and
> `agentfix solve` stops with a pointer to `exercises/README.md`. That is the intended starting
> point. For the finished agent: `git checkout solutions`.

Every exercise test runs against a scripted fake model, so the workshop does not depend on your
Ollama setup working. Real inference is the reward, not a prerequisite.

## Which setup option should you use?

| Option | Who | RAM | Model |
|---|---|---|---|
| 1 (default) | 16 GB+ laptop | 16 GB+ | Mellum2 12B via Ollama (~8 GB download) |
| 2 | weaker laptop | ~4 GB | `qwen2.5-coder:1.5b` (~1 GB) |
| 3 | browser only | any | Google Colab — `notebooks/agentfix.ipynb` |

Options 1 and 2 run on macOS, Linux, WSL2 and native Windows; per-OS commands are in the setup
section below. Option 3 needs only a browser.

**Windows users: prefer WSL2.** The sandbox that executes the agent's test runs is POSIX-shaped.

All measurements in this README were taken on macOS with Option 1 unless stated otherwise. Where a
path is untested, it says so.

## Setup

### Step 1 — install `uv` and Ollama

<details open>
<summary><b>macOS</b> (verified)</summary>

```bash
brew install uv ollama
ollama serve &                  # or: open -a Ollama   (the app starts the same server)
```

Homebrew's `ollama` and Ollama.app are the same server on `localhost:11434` — use either, not
both. Without Homebrew: `curl -LsSf https://astral.sh/uv/install.sh | sh` and Ollama from
[ollama.com/download](https://ollama.com/download).
</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh
```

The install script registers a systemd service, so the server is already listening;
`systemctl status ollama` confirms it. If you installed the tarball by hand, run `ollama serve` in
its own terminal. A GPU is not required — CPU inference works, just slower.
</details>

<details>
<summary><b>Windows — WSL2 (recommended)</b></summary>

In PowerShell, once:

```powershell
wsl --install -d Ubuntu
```

Then follow the **Linux** instructions inside the Ubuntu shell and do everything — `git clone`,
`uv`, `ollama`, the exercises — inside WSL2. Keep the clone on the Linux filesystem
(`~/agentfix-langchain`, not `/mnt/c/...`); test runs across the `/mnt/c` bridge are slow enough
to be annoying.

WSL2 takes a fraction of your RAM by default (50%, capped at 8 GB on older builds), and that
fraction — not your machine's spec sheet — has to hold an 8 GB model. If `free -g` inside WSL2
shows under 16 GB, raise it in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

then `wsl --shutdown` in PowerShell and reopen the shell.
</details>

<details>
<summary><b>Windows — native PowerShell</b> (exercises work; sandbox untested)</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --id Ollama.Ollama      # or the installer from ollama.com/download
```

The installer runs Ollama in the background, so the server is already on `localhost:11434` (look
for the tray icon). Every `uv run ...` command below is identical in PowerShell, and forward
slashes in task paths are fine.

Two caveats: `agentfix doctor` cannot read RAM on Windows and skips that check rather than failing
it, and the subprocess sandbox has not been run on native Windows. If `doctor` reports a `sandbox`
failure, switch to WSL2 rather than debugging it during the workshop.
</details>

### Step 2 — get the model

<details open>
<summary><b>Option 1 — Mellum2 (16 GB+ RAM)</b></summary>

```bash
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
ollama create agentfix-mellum2 -f Modelfile
```

The `create` step derives a model with `num_ctx 16384` baked in. Two things worth knowing: the
name `agentfix-mellum2` is what `DEFAULT_MODEL` in `src/agentfix/config.py` expects, and the
context setting is now belt-and-braces rather than essential — this edition talks to Ollama's
native API, which honours a per-request `num_ctx`. See [The context
window](#the-context-window).
</details>

<details>
<summary><b>Option 2 — the 1 GB fallback</b></summary>

```bash
ollama pull qwen2.5-coder:1.5b
export MELLUM_MODEL=qwen2.5-coder:1.5b     # PowerShell: $env:MELLUM_MODEL="qwen2.5-coder:1.5b"
```

No `ollama create` step: the client sends `num_ctx` with every request and Ollama's native API
honours it, so a plain pull is enough. Set `MELLUM_MODEL` in every shell you use, or put it in
your shell profile.

A 1.5B model fixes fewer bugs than Mellum2 and gets its tool-call JSON wrong more often. That is
not a broken setup — it is the reason the loop guard in Stage 2 exists. Untested by the author;
the exercises do not depend on it.
</details>

<details>
<summary><b>Option 3 — Google Colab</b></summary>

Open `notebooks/agentfix.ipynb` in Colab and run the cells in order. It installs Ollama, pulls
`qwen2.5-coder:1.5b`, clones this repo at `main`, disables pushing, and runs each stage's tests
from a cell. Everything else in this README applies, except that you edit files and run checks in
notebook cells rather than in an IDE.
</details>

### Step 3 — install and check

```bash
uv sync --extra dev
uv run agentfix doctor
```

`doctor` is the fastest way to find a broken setup, because almost every failure here produces a
symptom that looks like something else — a too-small context window looks like a stupid model, not
a misconfiguration. A healthy Option 1 machine reports:

```
[PASS] python: 3.13.14
[PASS] ram: 24.0 GB total, 4.0 GB free
[PASS] ollama installed: /usr/local/bin/ollama
[PASS] ollama server: reachable at http://localhost:11434
[PASS] model present: agentfix-mellum2
[PASS] generation: 37 tok/s (372 tokens in 10.1s)
[PASS] context window: 16384 tokens
[PASS] sandbox: executes tests
READY
```

## Build the agent

Two stages, both offline, both editing `src/agentfix/agent/graph.py`:

```bash
uv run python -m unittest exercises.stage_1.test_stage_1 -v
uv run python -m unittest exercises.stage_2.test_stage_2 -v
```

Instructions are in [`exercises/README.md`](exercises/README.md) and the per-stage READMEs. On a
fresh `main` clone both fail and `agentfix solve` does not work — that is the intended starting
point, not a broken checkout.

Stuck? `git checkout stage-1-solution` or `stage-2-solution` jumps ahead without falling behind
the room. The `solutions` branch is the finished agent.

## Use

```bash
uv run agentfix solve tasks/workshop/01-shopcart --verbose
uv run agentfix eval --suite workshop
uv run agentfix eval --suite humanevalfix --limit 5
```

## Tests

unittest only, no pytest anywhere — including inside the task fixtures the agent fixes.

```bash
uv run python -m unittest discover -s tests -t .          # 193 tests, offline, ~5s
uv run python -m unittest discover -s exercises -t .      # the 22 exercise tests
AGENTFIX_LLM_TESTS=1 uv run python -m unittest discover -s tests -t .   # + live-model tests
```

The whole suite runs with no model process anywhere: `llm/fake.py` is a real `BaseChatModel` that
returns a scripted list of replies, so the tests drive the **real** graph against the **real**
tools in a real temp directory. Only the model is replaced.

`uv sync --extra dev --extra prebuilt` additionally enables `tests/test_prebuilt.py`; without the
extra those tests skip.

## Reading order

1. `tools/base.py` — what a tool is, the limits on what it may return, and the artifact channel
2. `tasks/loader.py` — what a task is; the copy-to-tempdir context manager
3. `tools/fs.py` — `list_files`, `read_file`, `write_file`
4. `tools/tests_tool.py` — `run_tests`, the agent's only oracle
5. `agent/state.py` — what the graph carries between nodes, and the reducers that combine it
6. `agent/graph.py` — **the agent.** If you read one file, read this one.
7. `runner.py` — how the pieces are wired together

Then `agent/trace.py` (observability as a LangChain callback handler), `llm/` (the real and
scripted models), `sandbox/` (how tests are executed), `eval/` (measurement), and `doctor.py`.

`agent/prebuilt.py` is the argument rather than the implementation: the same agent built from
`langchain.agents.create_agent` and its middleware, with each claim carrying the measurement
behind it. Needs `--extra prebuilt`.

## What the framework gave us

- `ToolNode` replaces the hand-written `dispatch`, including its unknown-tool and bad-argument
  observations, and answering several calls in one turn.
- `add_messages` makes the history append-only by construction.
- Reducers on the rest of the state — `operator.add` for the token counters, a two-argument
  `keep_larger` for the peak — so a node returns a delta and never reads the old value.
- Callbacks carry the trace. `agent/trace.py` is a `BaseCallbackHandler` handed to the graph once,
  so the nodes contain no tracing code at all.
- Checkpointing: `InMemorySaver` snapshots the state after every node, which makes a run resumable
  and its history inspectable step by step.
- `ChatOllama` parses tool calls and token usage. Note what it does *not* do: malformed
  tool arguments are not rescued into a reply you can answer — they raise.

## What it did not

- **`handle_tool_errors` defaults to letting a tool's exception kill the run.** You have to opt
  back in — and passing a *string* rather than `True` silently discards the specific error, so the
  model stops being told which argument it forgot.
- **The loop guard.** LangGraph has no hook for it. LangChain 1.x gives you the seam
  (`wrap_tool_call`) but not the policy — which is Stage 2 of the workshop. It also leaves you
  the invariant the guard has to respect: every tool call needs exactly one reply, keyed by
  `tool_call_id`, so even a call you REFUSE to run still has to be answered. Skip one and the
  *next* request is rejected, a turn away from the code that caused it.
- **The step budget — on LangGraph.** `recursion_limit` counts node executions, not model turns.
  On LangChain 1.x this one has *moved*: `ModelCallLimitMiddleware(run_limit=N)` counts exactly
  what `AgentState.step` counts. See `agent/prebuilt.py`, including the measurement showing it is
  silently ignored if you order the middleware wrong.
- **Checkpointing is only as good as what you put in the state.** The test verdict used to live on
  the `run_tests` tool object. The graph was resumable; the agent was not — a resumed run rebuilt
  that tool empty and reported a solved task unsolved. The verdict now travels as a `ToolMessage`
  artifact into `AgentState.tests_passed`.
- **Tool calls in one turn run concurrently by default.** `ToolNode` batches through a real
  `ThreadPoolExecutor` even when nothing asked for it, so a `run_tests` in the same message as a
  `write_file` can measure the file as it was *before* the write. Message order is preserved, so
  the trace looks innocent. `max_concurrency=1` restores one-at-a-time execution.
- **The wrong integration will lie to you.** This repo used `ChatOpenAI` against Ollama's `/v1`
  endpoint to keep the wire format byte-identical to the no-framework original. Two of the three
  settings that decide whether the agent works were being discarded in transit, silently.
  Measured, same server:

  | | `ChatOpenAI` via `/v1` | `ChatOllama` |
  |---|---|---|
  | cap on one reply | `max_completion_tokens=8` → 692 tokens | `num_predict=8` → 8 tokens |
  | context window | `options` dropped; `ollama ps` says 4096 | `num_ctx=8192` → `ollama ps` says 8192 |

  A compatibility endpoint accepts the requests it does not honour.

## The context window

The single most consequential setting, and the one nothing else will tell you about. Too small a
window does not error — it silently truncates the middle of the agent's history, which looks like a
stupid model rather than a misconfigured one. `agentfix doctor` checks it against
`MIN_CONTEXT_LENGTH` and fails if the loaded model reports less.

Because this edition uses `ChatOllama`, the client sends `num_ctx` with every request and Ollama's
native API honours it — verified by asking for 8192 against a model derived at 16384 and watching
`ollama ps` report 8192. The `Modelfile` remains the documented path for Option 1 because it gives
the model a stable name, but it is no longer the only way to get a working context window.

## Measured performance

Option 1, macOS, 24 GB RAM, 37 tok/s:

```
task                     solved   steps   tokens    peak ctx   seconds
01-shopcart              True     8       8566      1387       26.05
02-invoice               True     8       9098      1574       30.38
03-parser                True     7       8322      1461       48.32
pass@1 = 1.00  (3 task(s))  peak prompt = 1574 tok
```

Eval is deliberately sequential, and that is measured rather than assumed: against this Ollama
server, three requests took 1.7s run one after another and 2.8s run concurrently. One local model
is one set of weights being time-shared.

## Parity with the no-framework edition

The port is meant to change the code and not the agent. Same task, same model:

| | no framework | LangGraph |
|---|---|---|
| verdict | SOLVED | SOLVED |
| steps | 8 | 8 |
| tokens | 8,626 | 8,646 |
| peak context | 1,388 | 1,384 |
| turns with reasoning | 0 of 7 | 0 of 7 |

Same tool sequence, step for step. The wire format is no longer byte-identical — that is the
deliberate trade described under **What it did not**.

**The agent does not reason.** Seven tool-calling turns, seven `(NO REASONING)` markers, and the
only prose arrives at step 8 *after* the fix is already verified. It also reads all three source
files despite being told not to, and finds the bug at step 4 but keeps looking. That is the
Act-only baseline from the ReAct paper, and closing that gap is the next act — the two functions
you write in the exercises are the two that have to change.

## Running things in Docker

The default sandbox is a hardened subprocess: stripped environment, resource limits, a timeout. It
is **not** a security boundary — test code runs as your user, on your machine. For real isolation:

```bash
docker build -t agentfix-sandbox -f Dockerfile.sandbox .
AGENTFIX_SANDBOX=docker uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

PowerShell wants `$env:AGENTFIX_SANDBOX="docker"` on its own line first. The container mounts the
workspace read-only, runs as a non-root user, and has no network. Note that `Dockerfile.sandbox`
installs nothing — `unittest` is in the standard library, so there is no version to pin and no
drift between the host and the container to catch.

Docker execution is untested by the author on this edition; the backend's own tests
(`tests/test_docker_backend.py`) assert the command line rather than starting containers, which is
what keeps them runnable everywhere.

## Platform notes

- **RAM check**: `doctor` reads available memory on macOS and Linux only. On Windows it skips the
  check rather than failing it.
- **The sandbox**: `subprocess_backend.py` uses POSIX resource limits. Untested on native Windows.
- **Case-insensitive filesystems**: macOS lets `Tests/TEST_CART.PY` address the same file as
  `tests/test_cart.py`, so the check protecting the agent's own oracle from the agent is
  deliberately case-insensitive. There is a reproduced-escape test for it.

## Known limitations

- One attempt per task, no retries and no best-of-n. `pass@1` means exactly that.
- The agent rewrites whole files rather than emitting diffs. At this model size a diff-based tool
  contract is one the model cannot satisfy, which looks exactly like a broken agent.
- Nothing stops the agent from writing code that special-cases the test inputs. The write
  allow-list and the protected test suite close the routes that were actually reproduced; that one
  stays open.
