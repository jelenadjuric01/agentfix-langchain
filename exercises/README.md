# Exercises

Two stages. Each one edits `src/agentfix/agent/graph.py` — the agent itself — and the tests run
**without a model**, so you can finish both offline and in any setup tier.

| Stage | You write | Test |
|---|---|---|
| 1 | `route_after_agent` — where a run can end, and on whose word | `uv run python -m unittest exercises.stage_1.test_stage_1 -v` |
| 2 | the loop guard — how a stuck model is caught | `uv run python -m unittest exercises.stage_2.test_stage_2 -v` |

Both are about the same question from two directions. Stage 1 is *when do we stop on purpose*.
Stage 2 is *when do we stop because the model is broken*. Everything else in this repo — the
tools, the sandbox, the state, the tracing — exists to make those two decisions possible.

They are also the two functions that have to be rewritten when this agent becomes a **ReAct**
agent, which is the sequel to this workshop. Write them here and that rewrite will feel like a
consequence rather than a lecture.

## Running the tests

    uv run python -m unittest exercises.stage_1.test_stage_1 -v      # stage 1
    uv run python -m unittest exercises.stage_2.test_stage_2 -v      # stage 2
    uv run python -m unittest discover -s exercises -t . -v          # both

The repo's own suite is a superset and will also go green as you go:

    uv run python -m unittest discover -s tests -t .

On a fresh `main` clone both stages fail, and `uv run agentfix solve ...` does not work. That is
the intended starting point, not a broken checkout.

## Stuck?

Jump ahead without falling behind the room:

    git checkout stage-1-solution     # stage 1 done, stage 2 still yours
    git checkout stage-2-solution     # both done — identical to the `solutions` branch

Or read one answer without moving your working tree:

    git diff main stage-1-solution -- src/agentfix/agent/graph.py

To get back to your own work: `git checkout main`.
