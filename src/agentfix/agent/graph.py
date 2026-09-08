"""The agent itself. If you read one file in this project, read this one.

The no-framework edition was a `for` loop. This is the same agent as a graph, and the three
things that decide whether it works at all are unchanged:

1. a bounded number of steps         — an agent with no cap is an unbounded wait and bill
2. a stop condition based on reality — `is_done`, which believes the test suite rather than
                                       the model's opinion of its own work
3. a loop guard                      — because a stuck model repeats itself forever

Those three are ours. Everything around them is the framework's, and the split is the point of
the whole repo.

What the framework does here:

  - `ToolNode` runs the calls. Dispatch, ordering, unknown tool names, argument validation and
    error recovery — one invocation per turn, handed the batch that survived the guard.
  - `add_messages` makes the history append-only by construction, which is what keeps the
    prompt prefix byte-stable and the server's KV cache valid.
  - The other reducers on `AgentState` accumulate the counters, so `agent_node` returns deltas
    and never reads the old value.
  - Callbacks carry the trace. No node below contains tracing code; the tracer is handed to
    the graph once in `run_agent`.
  - The checkpointer snapshots the state after every node, so a run can be resumed or
    inspected step by step.

What it does NOT do, and this is the part worth the workshop's time:

  - `handle_tool_errors` defaults to letting a tool's exception propagate and kill the run.
    The original guaranteed dispatch never raises. We opt back in, below.
  - The loop guard. The seams exist — `wrap_tool_call` in LangChain 1.x middleware, and a
    `post_model_hook` if you build the agent with `create_react_agent` — but a seam is only a
    place to put a decision, and the framework owns the state behind it. agent/prebuilt.py
    builds this very guard on `wrap_tool_call` and records the measured cost: its counters
    live on the middleware instance, so they survive no checkpoint and they leak into the
    next run. Here they are `AgentState` fields, scoped to the run because the state is.

    The POLICY is what is missing, and the graph below now says so by construction. It borrows
    the framework's own shape for the job: `create_react_agent` wires `post_model_hook` by
    diffing the answered `tool_call_id`s against the calls the model made and dispatching only
    what is left, so answering a call is what refuses it. The hook itself is a
    `create_react_agent` argument rather than a `StateGraph` one, but `Send` is public, so
    `guard_node` plus `route_after_guard` is that router, reproduced — and there is no
    synthetic AIMessage anywhere in this file as a result.


  - The step budget, here. `recursion_limit` counts node executions, not model turns. LangChain
    1.x ships `ModelCallLimitMiddleware`, which counts the right thing; agent/prebuilt.py uses
    it, and records the ordering trap that makes it silently do nothing.

And one thing it does not do that took a while to see: checkpointing is only as good as what
you put in the state. While the test verdict lived on the run_tests tool, the graph was
resumable and the agent was not — see `AgentState.tests_passed`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from agentfix.agent.state import AgentState, initial_state
from agentfix.agent.trace import Tracer, TraceEvent, prompt_tokens_of
from agentfix.sandbox.base import ExecResult
from agentfix.tasks.loader import Task
from agentfix.tools.base import WorkspaceChanged

# Raised to 10 from an original 6 after measurement: this tool granularity needs run_tests +
# list_files + one read_file per implicated file + write_file + a verifying run_tests, which is
# already 8 steps for a three-file read.
MAX_STEPS = 10

# Three identical calls in a row is a stuck model, not slow progress.
MAX_GUARD_HITS = 3

# Sent when the model replies with prose while the tests are still red. See `route_after_agent`
# for why a text-only reply is not allowed to end the run.
NUDGE = "The tests have not passed. Read the latest failure and write a fix."


@dataclass(frozen=True)
class AgentResult:
    """Everything one run produced: the verdict plus what it cost to reach it."""

    task_id: str
    solved: bool
    steps_used: int
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    trace: tuple[TraceEvent, ...]
    peak_prompt_tokens: int = 0

    # Set only when the RUN ITSELF failed — the model server went away, the graph was wired
    # wrong — as opposed to the agent failing to fix the bug. `solved=False` cannot tell those
    # apart, and the difference is the difference between a score and a broken harness. It is a
    # plain string rather than the exception so it survives into the eval JSON, which drops the
    # trace. See eval/runner.crashed.
    error: str | None = None


def system_prompt(tools: Sequence[BaseTool]) -> str:
    """The standing instructions, rebuilt from whatever tools are actually registered.

    The tool names are derived from the tools themselves rather than hardcoded, so adding a
    tool updates the prompt and the schema together and they cannot drift apart.

    Most of this text is the result of watching the model fail: it read every file in the
    project, it emitted diffs instead of whole files, it declared victory without re-running
    the tests. Each instruction below is a countermeasure to an observed failure.
    """
    names = ", ".join(tool.name for tool in tools)
    return (
        "You are a Python bug-fixing agent working in a small project.\n"
        f"You have these tools: {names}.\n"
        "Work in this order: run the tests to see what fails, read the relevant file(s) "
        "before editing, then write the corrected file.\n"
        "Only read files the failure actually implicates — do not read every file "
        "list_files returns.\n"
        "When you call write_file you must supply the COMPLETE file contents, not a diff.\n"
        "Then run the tests again to confirm the fix worked — you are not finished until "
        "they pass. If they still fail, read the new failure and try again.\n"
        "Make the smallest change that fixes the failure. Do not rewrite unrelated code."
    )


def task_prompt(task: Task) -> str:
    """The task's own prompt, verbatim. A seam: the agent never invents task text."""
    return task.prompt


def is_done(state: AgentState) -> bool:
    """The agent is done when the tests actually pass — never because it says so.

    The highest-value idea in the whole design, and the only thing that can end a run early.
    Two failure modes it rules out: a model that claims a fix it never made, and a model that
    passed the tests once and then broke them again — see `tests_passed_after` for how the
    second one is kept out.

    A one-line read of the state, and that is the point: the verdict is computed from the tool
    answers as they arrive, so nothing outside the graph has to be consulted or trusted.
    """
    return state["tests_passed"]


def tests_passed_after(replies: Sequence[AnyMessage], current: bool) -> bool:
    """Fold one turn's tool answers into the verdict, in the order they happened.

    Driven by artifact TYPE rather than tool name, so the rule is about what a tool did rather
    than what it is called: an `ExecResult` is evidence about the code as it stands, and a
    `WorkspaceChanged` means the code no longer stands as measured.

    Order matters within a single turn, which is why this is a fold and not a pair of ifs. A
    model that calls run_tests and then write_file in one message ends the turn red, and one
    that writes and then runs ends it with whatever the run said.

    Pass it THIS turn's replies, never the whole history. The checkpointer's serialiser
    round-trips these artifacts back as plain dicts, so the isinstance checks below hold only
    for messages this process just produced. The verdict itself is a bool in the state, which
    survives a checkpoint intact — which is the whole reason it is a bool in the state.
    """
    for message in replies:
        if not isinstance(message, ToolMessage):
            continue
        if isinstance(message.artifact, ExecResult):
            current = message.artifact.passed
        elif isinstance(message.artifact, WorkspaceChanged):
            # The tests may well still pass — but nothing has measured this code yet, and an
            # unmeasured guess is exactly what `is_done` exists to refuse.
            current = False
    return current


def call_signature(call: dict[str, Any]) -> str:
    """An identity for "the same call again": tool name plus its arguments.

    `sorted(...)` first so that {"a": 1, "b": 2} and {"b": 2, "a": 1} compare equal — key order
    in the model's JSON is not meaningful.
    """
    return f"{call['name']}::{sorted((call.get('args') or {}).items())!r}"


def requested_calls(message: AIMessage) -> list[tuple[dict[str, Any], str]]:
    """Every call the model made this turn, paired with its guard signature.

    Every call is assumed to carry an `id`, which is what a reply is paired with. Upstream types
    it `str | None`; ChatOllama always synthesises a uuid, so it is always there for us. A
    backend that omitted one would fail inside ToolMessage validation, and that is the right
    outcome — there is no correct reply to a call you cannot address.
    """
    return [(dict(call), call_signature(dict(call))) for call in message.tool_calls]


def guard_observation(name: str, hits: int) -> str:
    """The text sent back for a repeated call, escalating on the second repeat."""
    if hits == 1:
        return (
            f"You already called {name} with these exact arguments and got the result above. "
            "Try a different tool or different arguments."
        )
    # Second repeat: name the consequence. Being explicit that the run will be abandoned
    # measurably helps a small model break out of the pattern.
    return (
        f"You have now called {name} with identical arguments {hits + 1} times in a row and it "
        "was not executed. Call a different tool or use different arguments — read the file the "
        f"failure names, or call write_file with a fix. After {MAX_GUARD_HITS} repeats this run "
        "is abandoned."
    )


def completion_tokens_of(message: AIMessage) -> int:
    """Tokens the model generated this turn, or 0 if the server reported none."""
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    return int(usage.get("output_tokens", 0))


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    tracer: Tracer,
    max_steps: int = MAX_STEPS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Assemble the agent. Returns a compiled graph you can `.invoke(state)`.

    Everything the nodes need that is not in the state — the model, the tools, the tracer — is
    captured here by closure. That is the graph equivalent of the original's "every tool was
    constructed with the workspace already bound": no node takes a path or a client as an
    argument, so no node can be pointed at the wrong workspace.
    """
    bound = llm.bind_tools(list(tools))
    # `handle_tool_errors=True` and not a custom message, for a reason worth knowing.
    # ToolNode's DEFAULT catches argument-validation errors but lets anything else — a genuine
    # exception inside a tool — propagate and kill the run, which breaks the original's "a tool
    # crash must not end the run" guarantee. So it has to be set.
    #
    # But passing a *string* here replaces the error text for every failure alike, and that
    # throws away the specific part: "path: Field required" becomes a generic apology, and the
    # model no longer knows which argument it forgot. `True` keeps ToolNode's own message,
    # which names the tool and the problem. Measured cost of getting this wrong: the model
    # retries blind.
    tool_node = ToolNode(list(tools), handle_tool_errors=True)

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Ask the model what to do. The whole history is re-sent; models are stateless.

        No tracing and no timing in here: the tracer is a callback handler, so LangChain calls
        it around this `invoke` on its own. What is left is a state transition, which is all a
        node should be.
        """
        reply = bound.invoke(state["messages"])
        assert isinstance(reply, AIMessage)

        # Every value here is a DELTA, combined with what is already in the state by that
        # key's reducer: messages append, the counters add, the peak takes the maximum. No
        # `state[...] + x` anywhere, so this node cannot get the accumulation wrong.
        return {
            # The message object itself is passed back untouched, which is what keeps the
            # prefix byte-stable for the server's KV cache.
            "messages": [reply],
            "step": 1,
            "prompt_tokens": prompt_tokens_of(reply),
            "completion_tokens": completion_tokens_of(reply),
            "peak_prompt_tokens": prompt_tokens_of(reply),
        }

    def guard_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Refuse the calls that mean the model is stuck, by ANSWERING them.

        This is the framework's own shape for the job, in a hand-built graph. `route_after_guard`
        works out what still needs running by diffing the calls the model made against the
        `tool_call_id`s already answered — so answering a call here is what refuses it, and the
        surviving subset never has to be restated anywhere. `create_react_agent` wires its
        `post_model_hook` exactly this way. The hook itself is a `create_react_agent` argument
        rather than a `StateGraph` one, but `Send` is public, so the shape is ours to use.

        What no framework supplies is the policy: that a call identical to the last one means the
        model learned nothing from the result and re-running it would buy the same output for
        another step. That claim is about a small model on a three-file project, not about graphs.

        The guard's own memory — the previous call's identity and the repeat count — lives in
        `AgentState`, which is what makes it survive a checkpoint and stay scoped to this run.
        """
        # The oracle guarantee. Now the dispatch below is the GRAPH's, one task per call, so the only
        # place that can serialise it is the run config.
        # Which is exactly why it is checked here instead of documented and hoped for. Without
        # it, a turn calling write_file and run_tests together can have the tests measure the
        # file as it was BEFORE the write, and `fold_node` would then take that stale-but-green
        # ExecResult as the verdict — the precise "believe the tests, not the model" guarantee
        # this project is built on. A raise rather than an assert: `python -O` strips asserts,
        # and a false SOLVED is not the kind of thing to lose to an optimisation flag.
        if config.get("max_concurrency") != 1:
            raise RuntimeError(
                "the tool step fans out one task per call, so the run config must carry "
                "max_concurrency=1 or a turn's calls execute in parallel and run_tests can "
                "measure the workspace as it was before a write in the same turn; "
                f"got {config.get('max_concurrency')!r}"
            )

        message = state["messages"][-1]
        assert isinstance(message, AIMessage)

        replies: list[AnyMessage] = []
        signature = state["last_signature"]
        hits = state["guard_hits"]

        for call, current in requested_calls(message):
            name = str(call.get("name") or "unknown")

            if current == signature:
                hits += 1
                replies.append(
                    ToolMessage(
                        content=guard_observation(name, hits),
                        tool_call_id=call["id"],
                        name=name,
                    )
                )
                tracer.note("tool", name, f"guarded — identical call #{hits + 1} in a row")
                continue

            # Progress: reset the counter and remember this call as the new baseline. Note the
            # baseline advances for a call that is merely DISPATCHED, not one known to have
            # succeeded — same as before, and deliberate: a call that ran and failed is still
            # new information, and repeating it verbatim is still the model going in circles.
            hits = 0
            signature = current

        return {"messages": replies, "last_signature": signature, "guard_hits": hits}

    def fold_node(state: AgentState) -> dict[str, Any]:
        """The single writer of `tests_passed`, and the reason it can stay reducer-free.

        This node exists because of the fan-out. With one task per call, several tool tasks land
        in the same superstep, and two writes to one reducer-free key in one step is something
        LangGraph refuses outright — `InvalidUpdateError`. Folding here keeps one writer, which lets `tests_passed`
        stay a plain bool: state.py argues for that on its own merits, because a reducer is
        handed (current, incoming) and cannot tell "the suite went green" from "the workspace
        changed, so the verdict is void".

        It also carries the invariant the tool step has to uphold. Every call the model made must
        get exactly one reply, matched by `tool_call_id`; leave one unanswered and the API rejects
        the NEXT request, a turn away from the cause. Checked here because here is the first point
        where the whole turn is visible — the guard's refusals and the tools' answers together.
        """
        messages = state["messages"]
        # THIS turn's replies, and no more: everything appended after the last AIMessage. The
        # fold recognises artifacts by TYPE, and a checkpoint round-trip hands them back as plain
        # dicts — so folding the whole history would quietly stop recognising anything at all.
        last_ai = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], AIMessage)
        )
        asked = messages[last_ai]
        assert isinstance(asked, AIMessage)
        turn = messages[last_ai + 1 :]

        requested = [call["id"] for call in asked.tool_calls]
        answered = [m.tool_call_id for m in turn if isinstance(m, ToolMessage)]
        if len(answered) != len(requested):
            raise RuntimeError(
                f"the tool step answered {len(answered)} of {len(requested)} tool calls; "
                "every call the model made must get exactly one reply"
            )

        return {"tests_passed": tests_passed_after(turn, state["tests_passed"])}

    def nudge_node(state: AgentState) -> dict[str, Any]:  # noqa: ARG001
        """A text-only reply while the tests are red is not a stop condition."""
        return {"messages": [HumanMessage(content=NUDGE)]}

    def route_after_agent(state: AgentState) -> str:
        """Where to go after a model turn. The only place a run can end successfully."""
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)

        # Tool calls never end the run on their own — always execute them and loop back, so
        # the model can read the results. Note this skips the `is_done` check on purpose: the
        # check belongs on a turn where the model had nothing more to do.
        if message.tool_calls:
            return "guard"

        # Prose. This is the only place the run can end successfully — and it ends because the
        # tests pass, not because the model stopped calling tools.
        if is_done(state):
            return END
        if state["step"] >= max_steps:
            return END
        return "nudge"

    def route_after_guard(state: AgentState) -> str | list[Send]:
        """Dispatch whatever the guard let through. The framework's router, reproduced.

        `pending` is the calls the model made minus the ones already answered — which is the
        whole mechanism behind refusing by answering. Each one is `Send`-ed as its own task, so
        no message anywhere has to state the surviving subset.

        The diff is scoped to THIS TURN, and that is deliberately stricter than the framework's
        own version of this router, which collects answered ids from the whole message history
        and so quietly assumes `tool_call_id`s are globally unique. Real ones are — ChatOllama
        synthesises a uuid per call. But the ids are the MODEL's to choose, and a model that
        reuses one would have its second call silently treated as already answered and never
        run.
        When the guard refused everything, there is nothing to dispatch and no tool will run,
        but the turn still has to be folded and routed — so it goes straight to `fold`.
        """
        messages = state["messages"]
        last_ai = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], AIMessage)
        )
        asked = messages[last_ai]
        assert isinstance(asked, AIMessage)
        answered = {
            m.tool_call_id for m in messages[last_ai + 1 :] if isinstance(m, ToolMessage)
        }
        pending = [call for call in asked.tool_calls if call["id"] not in answered]
        if not pending:
            return "fold"
        # `type="tool_call"` selects ToolNode's per-call Send payload: it hydrates the state it
        # needs from the graph's channels instead of being handed an inlined snapshot.
        return [Send("tools", [dict(call, type="tool_call")]) for call in pending]

    def route_after_tools(state: AgentState) -> str:
        """Stop if the model is stuck or out of budget; otherwise take another turn.

        Note what is NOT here: `is_done`. A turn whose tools just went green does not end the
        run — the model gets one more turn, and `route_after_agent` ends it there. That is a
        deliberate choice rather than an oversight, and it is not free: measured on 01-shopcart,
        the closing prose turn cost 6.5s of a 19.1s run and ~50 tokens of context.

        What it buys is the closing statement itself. That turn is the only prose in a run, and
        it is the evidence for the claim in the README that this agent does not reason — seven
        tool-calling turns carrying no reasoning, and the one explanation arriving *after* the
        fix was already verified. Ending on the green test result would stop collecting the
        artifact that demonstrates the finding. It also keeps every trace the same shape, ending
        on an `llm` line whether the run succeeded or not, which is what makes two of them
        comparable side by side.

        Adding the check here would be sound, and for a production agent it is probably right:
        one model turn cheaper per solved task, and it would sharpen "done" from "the model had
        nothing left to do AND the tests pass" to just "the tests pass". It would also make
        `route_after_agent`'s `is_done` unreachable, since `tests_passed` only ever becomes true
        in `fold_node` — so success would live in exactly one place instead of one real place
        and one vestigial one.

        The extra turn is not a soundness hole either way: if the model spends it on a write, the
        fold clears the verdict and the run correctly carries on.
        """
        if state["guard_hits"] >= MAX_GUARD_HITS:
            return END
        if state["step"] >= max_steps:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("guard", guard_node)
    # ToolNode is added as a NODE now, rather than invoked by hand from inside one. Dispatch,
    # ordering, unknown tool names, argument validation and the `handle_tool_errors` recovery
    # are its job, and the graph drives it one call at a time through the Send above.
    graph.add_node("tools", tool_node)
    graph.add_node("fold", fold_node)
    graph.add_node("nudge", nudge_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["guard", "nudge", END])
    graph.add_conditional_edges("guard", route_after_guard, ["tools", "fold"])
    graph.add_edge("tools", "fold")
    graph.add_conditional_edges("fold", route_after_tools, ["agent", END])
    graph.add_edge("nudge", "agent")

    # With a checkpointer the graph writes a snapshot of the state after every node, keyed by
    # the `thread_id` in the run config. That buys resumption and, in a debugger, time travel:
    # `get_state_history(config)` hands back every step this run went through.
    #
    # Worth knowing that this only became honest once the verdict moved into the state. While
    # `tests_passed` lived on the run_tests tool, a resumed run rebuilt that tool empty and a
    # solved task came back unsolved — the graph was checkpointable and the agent was not.
    return graph.compile(checkpointer=checkpointer)


def run_agent(
    task: Task,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> AgentResult:
    """Run the agent until the tests pass, the step budget runs out, or it gets stuck.

    Note there is no `work_dir` parameter: every tool was constructed with the workspace
    already bound (see runner.py), so nothing here touches a path. And no `run_tests`
    parameter either — the verdict arrives in the state as the tools answer, so nothing here
    needs a handle on the oracle itself.
    """
    # `tracer or Tracer()` rather than a default argument of `Tracer()`: a mutable default is
    # evaluated once at function definition and would be shared by every call.
    tracer = tracer or Tracer()
    # One saver per run, so a second run of the same task cannot resume the first one's thread.
    app = build_graph(llm, tools, tracer, max_steps=max_steps, checkpointer=InMemorySaver())

    started = time.time()
    # `recursion_limit` is LangGraph's own backstop and counts SUPERSTEPS, not model turns. A
    # turn is now agent + guard + tools + fold, and sometimes a nudge as well — twice what it
    # was when the guard and the fold lived inside one tools node, which is why the multiplier
    # went up with them. Set generously, because the budget that actually matters is
    # `max_steps`, enforced in the routers above. Hitting this limit raises, which is the
    # correct behaviour for "the graph is wired wrong".
    final: AgentState = app.invoke(
        initial_state(system_prompt(tools), task_prompt(task)),
        config={
            # Not a performance knob. The tool step fans out one task per call, and these
            # tools are not independent: run_tests measures what write_file just wrote. One
            # worker keeps a turn's calls in the order the model asked for them, which is what
            # `guard_node` refuses to run without. See tests/test_hook_alternative.py.
            "max_concurrency": 1,
            # The tracer is handed to the framework here, once, and LangChain calls it around
            # every model and tool invocation inside the graph — including the ones ToolNode
            # makes on our behalf. This is why no node contains tracing code.
            "callbacks": [tracer],
            "recursion_limit": max_steps * 5 + 10,
            # Which conversation this is. One task, one thread — the checkpointer files every
            # snapshot under it, and resuming means invoking again with the same id.
            "configurable": {"thread_id": task.task_id},
        },
    )

    return AgentResult(
        task_id=task.task_id,
        # Read from the final state, not from how the graph exited: ending on MAX_GUARD_HITS
        # or running out of steps must not be mistaken for success.
        solved=is_done(final),
        steps_used=final["step"],
        prompt_tokens=final["prompt_tokens"],
        completion_tokens=final["completion_tokens"],
        duration_s=round(time.time() - started, 2),
        trace=tuple(tracer.events),
        peak_prompt_tokens=final["peak_prompt_tokens"],
    )
