# Stage 1 — When is the agent done?

Open `src/agentfix/agent/graph.py` and find `EXERCISE(stage-1)` in `route_after_agent`.

This function runs after every model turn and returns one of three destinations: `"tools"`,
`"nudge"`, or `END`. It is the only place in the whole graph where a run can end *successfully*.

## What to write

Three rules. The order they are checked in matters as much as the rules themselves.

1. **The model asked for tools.** Go to `"tools"`. Always — even if the tests already pass.
   A turn where the model still wants to act is not a turn on which to ask whether it has
   finished. Note that a call whose JSON arguments did not parse still counts as asking:
   `message.invalid_tool_calls` is as much a request as `message.tool_calls`.

2. **The model replied with prose and the tests pass.** Return `END`. This is the successful
   exit, and the only one.

3. **The model replied with prose and the tests do not pass.** Go to `"nudge"` — a node that
   appends a message telling it the tests are still red and sends it back for another turn.
   Unless the step budget is gone (`state["step"] >= max_steps`), in which case `END`.

Two things are already written for you: `is_done(state)` reads the verdict, and `NUDGE` is the
text the nudge node sends. You need neither the tools nor the model.

## The trap

The two tempting stop conditions are both wrong, and the tests will catch both:

- *"the model stopped calling tools"* — it may have given up, or hallucinated a fix
- *"the model said it was done"* — models say that about code that does not work

The agent is done when **the tests pass**. Verification by execution, not by assertion. That is
the difference between a demo and something you would let near real code, and it is the reason
`is_done` reads a value that only a real test run can set.

## Run it

    uv run python -m unittest exercises.stage_1.test_stage_1 -v

Then, once it is green, run the whole agent against a real bug:

    uv run agentfix solve tasks/workshop/01-shopcart --verbose

It will not solve it yet — nothing stops a stuck model until stage 2 — but you should see it
reach a passing test suite and stop there instead of talking its way out.

## Finished early?

Look at `route_after_tools`, directly below. It has the two *failure* exits and deliberately
does **not** check `is_done`. Its docstring explains why, and what the choice costs. Do you
agree with it?
