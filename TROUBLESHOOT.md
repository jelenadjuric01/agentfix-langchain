# Troubleshooting

Everything that has actually gone wrong for someone setting this repository up, and what fixed
it. Nothing here is a bug in your work — these are environment problems, and each one has a
short answer.

Click a heading to open it.

**Start with `doctor`.** Almost every failure below produces a symptom that looks like something
else — a too-small context window looks like a stupid model, not a misconfiguration — so before
reading further:

```bash
uv run agentfix doctor
```

It prints `[PASS]`/`[FAIL]` per check with the fix in the failure line. The
[Setup](README.md#setup) section of the README shows what a healthy machine reports.

## `uv`, the venv and the IDE

<details>
<summary><b><code>uv: command not found</code>, right after installing it</b></summary>

The installer put `uv` in `~/.local/bin` and appended that to your shell profile — which only
affects shells started afterwards.

```bash
source ~/.zshrc                  # or ~/.bashrc; or just open a new terminal
```

If it is still missing, `~/.local/bin/uv --version` tells you whether the install worked at all.
On Windows, see the Windows section below: the same rule applies, but restarting the terminal is
not always enough.
</details>

<details>
<summary><b><code>error: Failed to spawn: agentfix</code>, or <code>No such file or directory</code></b></summary>

The project has not been installed into its environment yet. `agentfix` is a console script
declared in `pyproject.toml`, so it does not exist until the package is installed:

```bash
uv sync --extra dev
uv run agentfix doctor
```

Run every command through `uv run`. A bare `agentfix doctor` looks for the script on your `PATH`,
where it is not — the venv is `.venv/` inside the project and `uv run` is what activates it for
one command.
</details>

<details>
<summary><b><code>ModuleNotFoundError: No module named 'langgraph'</code> (or <code>langchain_ollama</code>, <code>datasets</code>)</b></summary>

You are running Python from outside the project environment, or the environment is not synced.

```bash
uv sync --extra dev              # langgraph, langchain-core, langchain-ollama, plus the dev tools
uv sync --extra eval             # + datasets, only needed for `eval --suite humanevalfix`
uv sync --extra dev --extra prebuilt   # + langchain, which enables tests/test_prebuilt.py
```

`datasets` and `langchain` are deliberately optional: the workshop and its tests do not need
either, and `tests/test_prebuilt.py` skips itself when `prebuilt` is absent rather than failing.

If `uv sync` itself fails, check the Python it found — this project needs **3.12 or newer**.
`uv python install 3.12` gets one without touching your system Python.
</details>

<details>
<summary><b>The IDE underlines <code>agentfix</code> imports red</b></summary>

The code is fine. The IDE does not know that the package starts inside `src/`.

Right-click **`src`** in the Project view → **Mark Directory as** → **Sources Root**. The red
underlines disappear immediately.

While you are there, point the interpreter at the project's own environment:
**Settings** → **Project: …** → **Python Interpreter** → the existing `.venv` in the project
root. Neither change affects `uv run`, which never consulted the IDE in the first place.
</details>

<details>
<summary><b>Tests pass in the terminal but the IDE shows them failing (or the reverse)</b></summary>

Two different interpreters. `uv run` uses `.venv/` inside the project; the IDE uses whatever is
set in **Settings** → **Project: …** → **Python Interpreter**.

Point the IDE at `.venv`. To make a plain terminal match it without `uv run`:
`source .venv/bin/activate` (`.venv\Scripts\activate` on Windows).
</details>

## Windows

<details>
<summary><b>You installed Ollama or set <code>MELLUM_MODEL</code>, and the repo still cannot see it</b></summary>

**Open a new terminal — and if you are running it from inside an IDE, restart the IDE.**

This is the one that catches everyone. Windows hands a process its environment variables when the
process starts and never updates them afterwards. `setx`, the Ollama installer and the
**Environment Variables** dialog all write the variable for *future* processes; the terminal you
are standing in, and the IDE that launched it, keep the environment they started with.

Quick check, in a **new** terminal:

```powershell
echo $env:MELLUM_MODEL
ollama --version
```

`$env:MELLUM_MODEL="qwen2.5-coder:1.5b"` sets it for the current window only, which is the fastest
way to prove the model is fine.
</details>

<details>
<summary><b><code>running scripts is disabled on this system</code> when installing <code>uv</code></b></summary>

PowerShell's default execution policy. Use the form from the README, which carries the bypass on
the command itself:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

That flag applies to that one command and changes nothing permanently on your machine.
</details>

<details>
<summary><b><code>ollama</code> is not recognized as a command, right after installing it</b></summary>

Same cause as the variables above: the installer added Ollama to `PATH`, but only for processes
started afterwards. Open a new terminal, and restart the IDE if you are running it from there.
</details>

<details>
<summary><b><code>doctor</code> reports a <code>sandbox</code> failure on native Windows</b></summary>

Expected, and not worth debugging during a workshop. `sandbox/subprocess_backend.py` uses POSIX
resource limits and has not been run on native Windows. Switch to WSL2 — `wsl --install -d Ubuntu`,
then follow the Linux instructions inside the Ubuntu shell.

`doctor` also cannot read RAM on Windows. It passes the check rather than failing it, reporting
`[PASS] ram: could not read memory on this platform — check by hand`, so that line is not a
problem either.
</details>

## The model, the disk and the RAM

<details>
<summary><b>The 8 GB pull dies part-way through, or you do not have the space</b></summary>

Ollama does not check free space up front: it writes the blob, then the manifest, and fails
somewhere in between. A pull that died leaves a partial blob behind, and **re-running the same
`ollama pull` resumes it** — it does not start over.

If the space is genuinely not there, take Option 2 from the README, which needs about 1 GB:

```bash
ollama pull qwen2.5-coder:1.5b
export MELLUM_MODEL=qwen2.5-coder:1.5b     # PowerShell: $env:MELLUM_MODEL="qwen2.5-coder:1.5b"
```
</details>

<details>
<summary><b><code>doctor</code> says <code>model present: FAIL</code></b></summary>

The name the agent asks for is not a name Ollama has. `DEFAULT_MODEL` in `src/agentfix/config.py`
is `agentfix-mellum2`, and `MELLUM_MODEL` overrides it.

```bash
ollama list                      # what you actually have
echo "$MELLUM_MODEL"             # what you are asking for, if anything
```

On Option 1 the two commands from the README are both required — the pull gets the weights, the
create gives them the short name the code expects:

```bash
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M
ollama create agentfix-mellum2 -f Modelfile
```

On Option 2 there is no `create` step, so `MELLUM_MODEL` has to name the pulled model exactly —
including the `:1.5b` tag.
</details>

<details>
<summary><b><code>doctor</code> says <code>context window: 4096</code> instead of <code>16384</code></b></summary>

The `ollama create` step was skipped, so you are talking to the base model rather than the derived
one:

```bash
ollama create agentfix-mellum2 -f Modelfile
```

Do not skip it. At 4,096 tokens Ollama drops the *earliest* messages once the conversation grows
past the limit — and the earliest message is the system prompt telling the agent it is not done
until the tests pass. The symptom is an agent that seems to forget the task halfway through, which
looks like a stupid model rather than a misconfigured one.

On Option 2 this is belt-and-braces rather than essential: this edition talks to Ollama's native
API, which honours a per-request `num_ctx`. See [The context
window](README.md#the-context-window).
</details>

<details>
<summary><b>Connection refused on <code>localhost:11434</code>, or <code>ollama server: FAIL</code></b></summary>

Ollama is installed but the server is not running.

```bash
open -a Ollama                   # macOS, the app
brew services start ollama       # macOS, installed with Homebrew
sudo systemctl start ollama      # Linux
ollama serve                     # anywhere, including WSL2 without systemd
```

On macOS the Homebrew service and the tray app are the same server on the same port — use one, not
both at once.
</details>

<details>
<summary><b>The model is painfully slow, or the machine runs out of memory</b></summary>

Mellum2 is an 8 GB model and wants 16 GB of RAM to be comfortable. Take Option 2 from the README:
`qwen2.5-coder:1.5b`, about 1 GB. It fixes fewer bugs and gets its tool-call JSON wrong more
often — which is not a broken setup, it is what a 1.5B model does. Note that this edition has no
recovery path for malformed tool-call JSON: it surfaces as an `error:` line from `solve`, not as a
guarded turn.

Inside WSL2, check `free -g` before blaming the machine: WSL2 takes a fraction of your RAM by
default (50%, capped at 8 GB on older builds), and that fraction is what has to hold the model.
Raise it in `%UserProfile%\.wslconfig` and `wsl --shutdown`:

```ini
[wsl2]
memory=16GB
```
</details>

<details>
<summary><b>You also ran a sibling edition, and now everything crawls</b></summary>

Two 8 GB models resident at once. Ollama keeps the last model loaded for **five minutes** after
the final request, so moving straight from this repo to `agentfix-react` — a separate 8 GB
checkpoint that shares no blobs with this one — can put both in memory together.

```bash
ollama ps                        # what is loaded right now
ollama stop agentfix-mellum2     # or whichever one you are done with
```

Permanently, in the **server's** own environment: `OLLAMA_MAX_LOADED_MODELS=1`. The macOS menu-bar
app needs `launchctl setenv OLLAMA_MAX_LOADED_MODELS 1` and a restart of Ollama; a systemd install
needs it in `systemctl edit ollama`.
</details>

<details>
<summary><b>On Option 3 (Colab), <code>uv run agentfix doctor</code> fails locally</b></summary>

Expected. Nothing is broken.

On that option there is no Ollama, no model and no server on your laptop by design — they live in
the Colab runtime. Skip `doctor` locally; `notebooks/agentfix.ipynb` runs the same checks inside
Colab, and those are the ones that have to pass. The exercises still run on your own machine
either way, because their tests use a scripted fake model and need no model at all.
</details>

## Running the agent

<details>
<summary><b><code>agentfix solve</code> exits with <code>NotImplementedError</code> and points at <code>exercises/README.md</code></b></summary>

That is the intended starting point, not a broken checkout. `main` is the exercise branch: in
`src/agentfix/agent/graph.py`, `route_after_agent` is unwritten and the loop guard inside
`tools_node` is missing, so the exercise tests fail and `solve` cannot run.

```bash
uv run python -m unittest exercises.stage_1.test_stage_1 -v
uv run python -m unittest exercises.stage_2.test_stage_2 -v
git checkout solutions           # the finished agent, if you would rather read it
```
</details>

<details>
<summary><b>The agent prints <code>NOT SOLVED</code></b></summary>

Not necessarily your bug. Real models do not fix every task, and `qwen2.5-coder:1.5b` is
noticeably less reliable at multi-step tool use than Mellum2 — including getting its tool-call JSON
wrong, which this edition reports as an `error:` line rather than recovering from.

Read the `--verbose` trace before assuming your code is wrong. You are looking for the shape of a
working loop: the model calls `run_tests`, looks around with `list_files` / `read_file`, writes a
file, and runs the tests again. If that shape is there and the run simply ran out of steps, your
implementation is doing its job — run it again, or move on.

If instead the run ends while the tests are still red, or ends the moment the model says it is
done, that points at your `route_after_agent` rather than at the model.
</details>

<details>
<summary><b>The run ends early with the repeated-call guard, or <code>this run is abandoned</code></b></summary>

That is your Stage 2 loop guard doing its job, not an error. A model that calls the same tool with
identical arguments over and over has stopped learning from its results; the guard answers the
repeat instead of executing it, escalates on the second one, and ends the run on the third.

If it fires on *every* task, the observation is probably not reaching the model — check what your
guard appends to the history before it routes back, and compare against
`exercises/stage_2/test_stage_2.py`, which pins the expected behaviour exactly.
</details>

<details>
<summary><b>An exercise test fails and you cannot see why</b></summary>

Every exercise test runs against a scripted **fake** model, so the failure is deterministic and
has nothing to do with Ollama, your setup option, or the network. Run the one test on its own:

```bash
uv run python -m unittest exercises.stage_1.test_stage_1 -v
```

`src/agentfix/llm/fake.py` is a real `BaseChatModel` returning a scripted list of replies, so the
tests drive the real graph against the real tools in a real temp directory — only the model is
replaced. Reading the script the failing test uses is usually faster than re-reading your own
code: it tells you exactly which turn is being simulated.
</details>

<details>
<summary><b>The Docker sandbox does not run</b></summary>

The subprocess backend is the default and the repo works without Docker at all. For the container
version you need a running daemon **and** a built image:

```bash
docker info
docker build -t agentfix-sandbox -f Dockerfile.sandbox .
AGENTFIX_SANDBOX=docker uv run agentfix solve tasks/workshop/01-shopcart --verbose
```

PowerShell wants `$env:AGENTFIX_SANDBOX="docker"` on its own line first. A missing image shows up
at solve time as "Unable to find image", not at build time. Docker execution is untested by the
author on this edition.
</details>

## Nothing here matches

```bash
uv run agentfix doctor
```

Read it top to bottom. Every failure line carries the remedy command, which is faster than
guessing, and the README's [Setup](README.md#setup) section explains what each line means.

Taking all of this back off your machine afterwards: [CLEANUP.md](CLEANUP.md).
