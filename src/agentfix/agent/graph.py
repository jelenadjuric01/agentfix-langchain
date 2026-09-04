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
  - The loop guard. LangGraph has no hook for it at all. LangChain 1.x middleware does give
    you the seam (`wrap_tool_call`), but the policy — a repeated call means the model is stuck
    — is still yours. See agent/prebuilt.py.
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

    def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Guard the model's calls, then let ToolNode run whatever survives.

        Two responsibilities, deliberately in this order. The guard is ours — no framework
        knows that a repeated call means the model is stuck. Executing the rest is ToolNode's,
        and it gets the whole surviving batch in one invocation rather than being driven one
        call at a time: dispatch, ordering, unknown tool names, argument validation and the
        `handle_tool_errors` recovery are all its job, and it is better at them than a loop
        here would be.

        The API requires an answer to every call the model made — skip one `tool_call_id` and
        the next request is rejected — so every branch below produces exactly one message per
        call, including the branches where nothing ran.
        """
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)

        replies: list[AnyMessage] = []
        runnable: list[dict[str, Any]] = []
        signature = state["last_signature"]
        hits = state["guard_hits"]

        for call, current in requested_calls(message):
            name = str(call.get("name") or "unknown")

            # Loop guard. A model that repeats a call verbatim learned nothing from the result,
            # so re-running it would burn a step for the same output. Send an observation
            # instead and let it try something else.
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

            # Progress: reset the counter and remember this call as the new baseline.
            hits = 0
            signature = current

            runnable.append(call)

        if runnable:
            # One invocation for the whole turn. The synthetic message exists because ToolNode
            # reads its calls off the last message, and the original may contain calls the
            # guard just refused — the surviving subset has to be stated somewhere.
            batch = AIMessage(content="", tool_calls=runnable)
            # `max_concurrency=1` is not a performance knob, it is the oracle guarantee.
            #
            # ToolNode runs a batch through `get_executor_for_config`, which is a real
            # ThreadPoolExecutor even when no concurrency was asked for — so the calls in one
            # turn execute in PARALLEL by default. Measured with a slow write and a fast check
            # in one message: the check started and finished while the write was still in
            # flight. Message order is preserved either way, so the trace looks innocent.
            #
            # That is a false-SOLVED waiting to happen. A turn calling write_file and run_tests
            # together could have the tests measure the file as it was BEFORE the write, and
            # the fold below would then take that stale-but-green ExecResult as the verdict —
            # the exact "believe the tests, not the model" guarantee this project is built on.
            #
            # One worker restores the original loop's one-call-at-a-time execution while
            # keeping the single batched invocation. Merged into the ambient config rather than
            # replacing it: LangGraph injects runtime keys that ToolNode requires, and passing
            # a bare dict here fails with "Missing required config key".
            produced = tool_node.invoke(
                {"messages": [batch]}, config={**config, "max_concurrency": 1}
            )["messages"]
            # The invariant this whole node exists to uphold, and a raise rather than an
            # assert on purpose. This is not a type narrowing — it is a claim about a third
            # party's behaviour across versions, and `python -O` strips asserts. Losing it
            # means an unanswered `tool_call_id`, which the API rejects on the NEXT request,
            # one turn away from the code that caused it.
            if len(produced) != len(runnable):
                raise RuntimeError(
                    f"ToolNode answered {len(produced)} of {len(runnable)} tool calls; "
                    "every call the model made must get exactly one reply"
                )
            replies.extend(produced)

        return {
            "messages": replies,
            "last_signature": signature,
            "guard_hits": hits,
            "tests_passed": tests_passed_after(replies, state["tests_passed"]),
        }

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
            return "tools"

        # Prose. This is the only place the run can end successfully — and it ends because the
        # tests pass, not because the model stopped calling tools.
        if is_done(state):
            return END
        if state["step"] >= max_steps:
            return END
        return "nudge"

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
        in `tools_node` — so success would live in exactly one place instead of one real place
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
    graph.add_node("tools", tools_node)
    graph.add_node("nudge", nudge_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "nudge", END])
    graph.add_conditional_edges("tools", route_after_tools, ["agent", END])
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
    # `recursion_limit` is LangGraph's own backstop and counts NODE executions, not model
    # turns: a turn is agent + tools, and sometimes a nudge as well. Set generously, because
    # the budget that actually matters is `max_steps`, enforced in the routers above. Hitting
    # this limit raises, which is the correct behaviour for "the graph is wired wrong".
    final: AgentState = app.invoke(
        initial_state(system_prompt(tools), task_prompt(task)),
        config={
            # The tracer is handed to the framework here, once, and LangChain calls it around
            # every model and tool invocation inside the graph — including the ones ToolNode
            # makes on our behalf. This is why no node contains tracing code.
            "callbacks": [tracer],
            "recursion_limit": max_steps * 3 + 10,
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
