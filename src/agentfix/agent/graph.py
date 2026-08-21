"""The agent itself. If you read one file in this project, read this one.

The no-framework edition was a `for` loop. This is the same agent as a graph, and the three
things that decide whether it works at all are unchanged:

1. a bounded number of steps         — an agent with no cap is an unbounded wait and bill
2. a stop condition based on reality — `is_done`, which runs the tests rather than asking the
                                       model whether it is finished
3. a loop guard                      — because a stuck model repeats itself forever

What the framework gave us: `ToolNode`, which absorbs the hand-written dispatch, its unknown-
tool and bad-argument observations, and answering several calls in one turn. Plus state,
checkpointing and streaming for free.

What it did NOT give us, and this is the part worth the workshop's time:

  - `handle_tool_errors` defaults to letting a tool's exception propagate and kill the run.
    The original guaranteed dispatch never raises. We opt back in, below.
  - `invalid_tool_calls` — a tool call whose JSON arguments did not parse — is silently
    ignored by ToolNode, producing NO reply message. The API requires an answer to every
    call the model made, so this is not a cosmetic gap. A 12B model gets that JSON wrong
    often enough to matter, so `_answer_invalid` below handles it by hand.
  - the loop guard. No framework knows that a repeated call means your model is stuck.

ARCHITECTURE.md walks through this same graph with the design decisions annotated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agentfix.agent.state import AgentState, initial_state
from agentfix.agent.trace import Tracer, TraceEvent
from agentfix.tasks.loader import Task
from agentfix.tools.tests_tool import RunTestsTool

# Raised to 10 from an original 6 after measurement: this tool granularity needs run_tests +
# list_files + one read_file per implicated file + write_file + a verifying run_tests, which is
# already 8 steps for a three-file read.
MAX_STEPS = 10

# Three identical calls in a row is a stuck model, not slow progress.
MAX_GUARD_HITS = 3

# Sent when the model replies with prose while the tests are still red. See `route_after_agent`
# for why a text-only reply is not allowed to end the run.
NUDGE = "The tests have not passed. Read the latest failure and write a fix."

INVALID_ARGUMENTS_MESSAGE = (
    "Your arguments for {name} were not valid JSON, so nothing ran. Send the call again "
    "with a single well-formed JSON object."
)


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


def is_done(run_tests: RunTestsTool) -> bool:
    """The agent is done when the tests actually pass — never because it says so.

    The highest-value idea in the whole design, and the only thing that can end a run early.
    Two failure modes it rules out: a model that claims a fix it never made, and a model that
    passed the tests once and then broke them again (`RunTestsTool.invalidate` clears
    `last_result` on every write, so a stale green result cannot be reused).
    """
    return run_tests.last_result is not None and run_tests.last_result.passed


def call_signature(call: dict[str, Any]) -> str:
    """An identity for "the same call again": tool name plus its arguments.

    `sorted(...)` first so that {"a": 1, "b": 2} and {"b": 2, "a": 1} compare equal — key order
    in the model's JSON is not meaningful.
    """
    return f"{call['name']}::{sorted((call.get('args') or {}).items())!r}"


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


def describe(message: AIMessage) -> str:
    """One line summarising a model turn, for the trace.

    Unlike the original, the model's prose is shown even when it also called a tool. That text
    is the only place the model's reasoning could appear, and discarding it meant never being
    able to answer "does this agent reason?" — measured answer, on Mellum2: no, it does not.
    """
    text = (message.text or "").strip()
    if message.tool_calls or message.invalid_tool_calls:
        # `or "unknown"`: an invalid tool call may not even have a parseable name.
        names = ", ".join(
            call["name"] or "unknown" for call in [*message.tool_calls, *message.invalid_tool_calls]
        )
        return f"calls {names}" + (f" -- {text}" if text else " -- (NO REASONING)")
    return text


def _usage(message: AIMessage) -> tuple[int, int]:
    """(prompt, completion) tokens for one turn, defaulting to 0 when unreported."""
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    run_tests: RunTestsTool,
    tracer: Tracer,
    max_steps: int = MAX_STEPS,
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
        """Ask the model what to do. The whole history is re-sent; models are stateless."""
        step = state["step"] + 1
        started = time.time()
        reply = bound.invoke(state["messages"])
        assert isinstance(reply, AIMessage)

        prompt, completion = _usage(reply)
        tracer.record(
            TraceEvent(
                step=step,
                kind="llm",
                name="assistant",
                detail=describe(reply),
                prompt_tokens=prompt,
                latency_s=round(time.time() - started, 2),
            )
        )
        return {
            # Appended, not replaced — see the `add_messages` reducer on AgentState. The
            # message object itself is passed back untouched, which is what keeps the prefix
            # byte-stable for the server's KV cache.
            "messages": [reply],
            "step": step,
            "prompt_tokens": state["prompt_tokens"] + prompt,
            "completion_tokens": state["completion_tokens"] + completion,
            "peak_prompt_tokens": max(state["peak_prompt_tokens"], prompt),
        }

    def _answer_invalid(call: dict[str, Any], step: int, tokens: int) -> ToolMessage:
        """A tool call whose JSON did not parse. ToolNode ignores these entirely.

        Naming the real problem matters: "missing argument: path" would be misleading, because
        the model probably did send a path — inside broken JSON.
        """
        name = call.get("name") or "unknown"
        tracer.record(
            TraceEvent(step, "tool", name, "invalid JSON arguments — not executed", tokens, 0.0)
        )
        return ToolMessage(
            content=INVALID_ARGUMENTS_MESSAGE.format(name=name),
            tool_call_id=call["id"],
            name=name,
            status="error",
        )

    def tools_node(state: AgentState) -> dict[str, Any]:
        """Run the model's calls, one at a time, guarding repeats and answering bad JSON.

        The API requires an answer to every call the model made — skip one `tool_call_id` and
        the next request is rejected — so every branch below produces exactly one message per
        call, including the branches where nothing ran.
        """
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)
        step = state["step"]
        # The context size of the turn that asked for these calls, so a tool line in the trace
        # reports the same `ctx` as the model line above it.
        turn_prompt_tokens = _usage(message)[0]

        replies: list[AnyMessage] = []
        signature = state["last_signature"]
        hits = state["guard_hits"]

        # Bad JSON first, so a turn mixing valid and invalid calls still answers all of them.
        # These go through the SAME guard as valid calls, which restores parity with the
        # no-framework edition: there, malformed arguments arrived as a normal call whose
        # `arguments` dict held a sentinel, so `_call_signature` covered them for free. Here
        # they arrive on a separate field and would otherwise be exempt — letting a model stuck
        # on one malformed call retry until the whole step budget was gone, instead of being
        # abandoned after three repeats.
        for bad in message.invalid_tool_calls:
            name = bad.get("name") or "unknown"
            current = f"{name}::INVALID::{bad.get('args')!r}"
            if current == signature:
                hits += 1
                replies.append(
                    ToolMessage(
                        content=guard_observation(name, hits),
                        tool_call_id=bad["id"],
                        name=name,
                        status="error",
                    )
                )
                tracer.record(
                    TraceEvent(
                        step,
                        "tool",
                        name,
                        f"guarded — identical invalid call #{hits + 1} in a row",
                        turn_prompt_tokens,
                        0.0,
                    )
                )
                continue
            hits = 0
            signature = current
            replies.append(_answer_invalid(dict(bad), step, turn_prompt_tokens))

        for call in message.tool_calls:
            current = call_signature(dict(call))

            # Loop guard. A model that repeats a call verbatim learned nothing from the result,
            # so re-running it would burn a step for the same output. Send an observation
            # instead and let it try something else.
            if current == signature:
                hits += 1
                replies.append(
                    ToolMessage(
                        content=guard_observation(call["name"], hits),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                tracer.record(
                    TraceEvent(
                        step,
                        "tool",
                        call["name"],
                        f"guarded — identical call #{hits + 1} in a row",
                        turn_prompt_tokens,
                        0.0,
                    )
                )
                continue

            # Progress: reset the counter and remember this call as the new baseline.
            hits = 0
            signature = current

            started = time.time()
            # One call at a time so the observations come back in the order the model asked,
            # and so a guarded call can be skipped without confusing ToolNode's bookkeeping.
            single = AIMessage(content="", tool_calls=[call])
            produced = tool_node.invoke({"messages": [single]})["messages"]
            # The invariant this whole node exists to uphold. An empty list here would leave
            # `call["id"]` unanswered and the API would reject the next request — so say so
            # now, rather than emitting a blank trace line and failing confusingly later.
            assert produced, f"ToolNode answered {call['name']} with no message"
            replies.extend(produced)
            tracer.record(
                TraceEvent(
                    step=step,
                    kind="tool",
                    name=call["name"],
                    detail=str(produced[0].content),
                    prompt_tokens=turn_prompt_tokens,
                    latency_s=round(time.time() - started, 2),
                )
            )

        return {"messages": replies, "last_signature": signature, "guard_hits": hits}

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
        if message.tool_calls or message.invalid_tool_calls:
            return "tools"

        # Prose. This is the only place the run can end successfully — and it ends because the
        # tests pass, not because the model stopped calling tools.
        if is_done(run_tests):
            return END
        if state["step"] >= max_steps:
            return END
        return "nudge"

    def route_after_tools(state: AgentState) -> str:
        """Stop if the model is stuck or out of budget; otherwise take another turn."""
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
    return graph.compile()


def run_agent(
    task: Task,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    run_tests: RunTestsTool,
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> AgentResult:
    """Run the agent until the tests pass, the step budget runs out, or it gets stuck.

    Note there is no `work_dir` parameter: every tool was constructed with the workspace
    already bound (see runner.py), so nothing here touches a path.

    `run_tests` is passed separately as well as being inside `tools` — `is_done` needs to ask
    it directly, rather than through the model.
    """
    # `tracer or Tracer()` rather than a default argument of `Tracer()`: a mutable default is
    # evaluated once at function definition and would be shared by every call.
    tracer = tracer or Tracer()
    app = build_graph(llm, tools, run_tests, tracer, max_steps=max_steps)

    started = time.time()
    # `recursion_limit` is LangGraph's own backstop and counts NODE executions, not model
    # turns: a turn is agent + tools, and sometimes a nudge as well. Set generously, because
    # the budget that actually matters is `max_steps`, enforced in the routers above. Hitting
    # this limit raises, which is the correct behaviour for "the graph is wired wrong".
    final: AgentState = app.invoke(
        initial_state(system_prompt(tools), task_prompt(task)),
        config={"recursion_limit": max_steps * 3 + 10},
    )

    return AgentResult(
        task_id=task.task_id,
        # Re-checked here rather than trusting how the graph exited: ending on MAX_GUARD_HITS
        # or running out of steps must not be mistaken for success.
        solved=is_done(run_tests),
        steps_used=final["step"],
        prompt_tokens=final["prompt_tokens"],
        completion_tokens=final["completion_tokens"],
        duration_s=round(time.time() - started, 2),
        trace=tuple(tracer.events),
        peak_prompt_tokens=final["peak_prompt_tokens"],
    )
