# Stage 2 — Catching a stuck model

Open `src/agentfix/agent/graph.py` and find `EXERCISE(stage-2)` in `guard_node`: the guard block,
the decision to refuse a call instead of running it.

## What to write

The identity of a call is already written for you. `call_signature(call)` returns a string that is
equal for two calls the model should not be allowed to repeat — the tool name plus its arguments,
with the keys sorted so that `{"path": "a.py", "content": "x"}` and
`{"content": "x", "path": "a.py"}` come out the same. Key order in the model's JSON is not
meaningful, and a signature that treated them as different would be a guard a model can walk
straight past.

**The guard block** sits inside the loop over `requested_calls(message)`. For each call you have
`current` (this call's signature), `signature` (the previous executed call's), and `hits` (how
many times it has repeated). You decide:

- Identical to the last one → do **not** run it. Count the repeat, and append exactly one
  `ToolMessage` carrying `guard_observation(name, hits)` so the model learns why nothing
  happened. Then move on to the next call.
- Otherwise → this is progress. Reset the count and remember this signature as the new
  baseline, then let the call through to be executed.

`guard_observation` is written for you too, including the escalation on the second repeat.

## The rule you must not break

**Every call the model made needs exactly one reply.** The API pairs each answer to its question
by `tool_call_id`, and a request that leaves one unanswered is rejected outright — so the branch
where you refuse to run something still has to produce a message. This is the single most common
way to get this stage subtly wrong: the guard works, and then the *next* request fails for
reasons that appear to have nothing to do with it.

## Why this is not the framework's job

`ToolNode` will dispatch calls, validate arguments, recover from tool exceptions and handle
unknown tool names. It will not do this. No framework knows that a repeated call means *your*
model has stopped learning from what it is told — that is a fact about a 12B model on a
three-file project, and facts like that stay in your code.

Seams for it do exist, and it is worth knowing where, because sooner or later you will go looking.
LangChain 1.x middleware gives you `wrap_tool_call` (see `agent/prebuilt.py`); `create_react_agent`
takes a `post_model_hook` that runs after the model has proposed its calls; LangChain even ships a
`ToolCallLimitMiddleware` that blocks over-budget calls and answers each one with a `ToolMessage`.

None of them is the answer. That middleware counts calls per tool *name* and never inspects the
arguments, so it cannot tell three reads of three files from three reads of the same file — a
budget, not a repetition detector. The other two are places to put a decision, and the policy is
still yours to write. Which is the whole point of the workshop: the framework absorbed the
plumbing and left you the judgement.

The seams also cost something, which is the second reason this belongs in your node.
`agent/prebuilt.py`'s guard keeps its counters on the middleware instance: they survive no
checkpoint, and they leak into the next run. In `AgentState` they are scoped to the run, because
the state is.

`post_model_hook` is the one worth answering properly, because it looks like a fit. Its router
diffs the tool calls the model made against the `tool_call_id`s already answered and dispatches
only what is left — so answering a call inside the hook is precisely what refuses it. That is the
guard contract, handed to you — and it is the shape this agent now uses, which is why there is no
synthetic `AIMessage` anywhere in `graph.py`. Two things stood in the way, and only the second
cost anything:

1. It is a `create_react_agent` argument, not a `StateGraph` one, so you cannot pass it here. But
   the *shape* is reproducible in a hand-built graph in about four lines — `Send` is public.
2. That shape sends one task per call, which makes the tool step a **concurrent writer**, and
   `AgentState.tests_passed` has no reducer. Two tool calls in one turn and it raises
   `InvalidUpdateError: At key 'tests_passed': Can receive only one value per step`.

Both halves are measured, so you can check rather than take it on faith:

    uv run python -m unittest tests.test_hook_alternative -v

The same file also settles the obvious worry in the other direction: `max_concurrency=1` throttles
the fan-out as well, so the oracle guarantee survives one task per call — though only from the run
config, which is why `guard_node` refuses to dispatch without it. What did not survive is the
single-writer verdict: the fold moved out to `fold_node`, rather than give the key a reducer that
`state.py` argues against on its own merits.

Which is the honest shape of the choice, and worth seeing once: the framework does not lack a
place to put this. It lacks the judgement, and it owns the state you would need to keep.

## Run it

    uv run python -m unittest exercises.stage_2.test_stage_2 -v

Then the real thing, end to end:

    uv run agentfix solve tasks/workshop/01-shopcart --verbose
    uv run agentfix solve tasks/workshop/02-invoice --verbose

Task 02's bug is not in the file its failing test names, which is why `list_files` and
`read_file` earn their place.

## Finished early?

Your guard watches **actions**. Ask what happens to it when the model is asked to *think* before
acting — a turn with reasoning and no tool call. Nothing in `call_signature` can see that turn,
and after stage 1 the router no longer treats prose as an ending. That hole is where the sequel
to this workshop starts.
