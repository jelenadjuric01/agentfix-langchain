# Instructor Runsheet

Two stages, ~90 minutes. The one segment to protect at all costs is Stage 1 — if the room runs
slow, cut everything else before it.

## Pre-workshop checklist

Run all of this on the machine you will present from, the day before:

```bash
uv sync --extra dev --extra prebuilt
uv run agentfix doctor                                    # every check PASS
uv run python -m unittest discover -s tests -t .          # 193 tests, ~5s
git checkout solutions && uv run agentfix solve tasks/workshop/01-shopcart --verbose
git checkout main                                         # back to the student state
```

- **Warm the model.** The first request after a cold start pays the load cost; a live demo that
  hangs for a minute reads as a broken repo. Run one `solve` before the room arrives.
- **Confirm `main` is stubbed.** `uv run python -m unittest discover -s exercises -t .` should
  fail. If it passes, you are on the wrong branch.
- Close other applications. `doctor` reports free RAM, and an 8 GB model on a 16 GB laptop with a
  browser open is where "it's just slow" comes from.

## Minute-by-minute

| Time | Segment | Command | Say |
|---|---|---|---|
| 0:00–0:10 | Setup triage; what an agent is | `uv run agentfix doctor` | An agent is a loop around a chat model that can call functions and see the results. Nothing more magical. Three things decide whether it works: a bounded number of steps, a stop condition based on reality, and a way to notice the model is stuck. |
| 0:10–0:20 | Live demo: the finished agent | `git checkout solutions && uv run agentfix solve tasks/workshop/01-shopcart --verbose` | **Run from `solutions`, not `main`.** `main` is deliberately stubbed. Read the trace out loud — one line per model turn, one per tool call, context size growing. Point at `(NO REASONING)` on every acting turn. |
| 0:20–0:32 | Tools, and the oracle | walk `src/agentfix/tools/tests_tool.py` and `tools/fs.py` | `args_schema` → the model emits a tool call → `ToolNode` runs it → a `ToolMessage` with the same `tool_call_id` → appended to history. Then the important half: `run_tests` returns its `ExecResult` as a message **artifact**, and that artifact is the only thing the stop condition believes. |
| 0:32–0:55 | **Stage 1** — `route_after_agent` | `git checkout main`; `uv run python -m unittest exercises.stage_1.test_stage_1 -v` | Students edit `src/agentfix/agent/graph.py`. The two wrong answers — "it stopped calling tools" and "it said it was done" — are both in the tests, so they will meet them rather than be warned about them. Checkpoint: `git checkout stage-1-solution`. **Protect this segment.** |
| 0:55–1:15 | **Stage 2** — the loop guard | `uv run python -m unittest exercises.stage_2.test_stage_2 -v` then `uv run agentfix solve tasks/workshop/02-invoice --verbose` | The classic bugs: forgetting that a refused call still needs a reply, and building a signature that key reordering defeats. Both are named directly in the tests. Task 02's bug is not in the file its failing test points at, which is why `list_files`/`read_file` matter. Checkpoint: `stage-2-solution`. |
| 1:15–1:25 | What the framework did and did not give you | `README.md`, then `src/agentfix/agent/prebuilt.py` | The two things they just wrote are the two no framework will do for them. Then show the middleware version: LangChain 1.x *does* now ship the step budget, and the ordering trap that makes it silently do nothing is the best five minutes in the repo. |
| 1:25–1:30 | Where this goes next | `results/precomputed/workshop.json` | Seven acting turns, zero reasoning. That is the Act-only baseline from the ReAct paper. The sequel rewrites exactly the two functions they wrote today, for one reason: once you ask the model to think, prose stops meaning "finished". |

## Cut order when the room runs slow

1. Drop the `prebuilt.py` segment (1:15–1:25) — it is the most interesting and the least load-bearing.
2. Drop the Docker/sandbox discussion if you had planned one.
3. Compress Stage 2 to `call_signature` only and hand out `stage-2-solution` for the guard block.
4. Never cut Stage 1. A student who leaves having written the stop condition has the point of the
   whole workshop.

## Checkpoint-tag rescue

A student who is stuck should not sit out the next segment:

```bash
git checkout stage-1-solution     # stage 1 done, stage 2 still theirs
git checkout stage-2-solution     # both done
git checkout main                 # back to their own work
```

Reading one answer without moving their tree:

```bash
git diff main stage-1-solution -- src/agentfix/agent/graph.py
```

Warn them that `git checkout` with uncommitted changes to the same file will refuse. Tell them to
`git stash` first, or to commit on a scratch branch.

## Do not run a live eval

`uv run agentfix eval --suite workshop` takes about 105 seconds on a warm 37 tok/s machine for
three tasks, which is survivable but dull. The HumanEvalFix suite is far longer. Show
`results/precomputed/workshop.json` instead and talk about what the columns mean: an agent that
solves a task in 8 steps and 8.5k tokens is not the same product as one that takes 4 and 8k.

## Real failure modes worth having ready

These come up on a 12B model and are the most useful things in the room:

- **Reads every file it was told not to read.** Mellum2 reads all three source files on task 01
  despite an explicit instruction. Prompt instructions are requests, not constraints.
- **Finds the bug and keeps looking.** On task 01 the failure is visible at step 4; the fix arrives
  at step 6. There is no planning step, so nothing notices that the work is done.
- **Malformed tool-call JSON.** Frequent enough at this size that `invalid_tool_calls` handling is
  not defensive programming — without it the run dies on the *next* request, one turn away from the
  cause.
- **Claims a fix it never made.** The reason Stage 1 exists. Demonstrate it by checking out `main`
  and running `solve`: with no stop condition the agent talks confidently and solves nothing.
