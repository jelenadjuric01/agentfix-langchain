"""The agent's state: what travels between the graph's nodes.

In the no-framework edition all of this lived in local variables inside one `for` loop. Making
it an explicit, typed structure is the single biggest change the framework asks of you, and it
buys two things: any node can read the whole state without arguments being threaded through,
and the state can be checkpointed, so a run can be resumed or inspected mid-flight.

It costs something too. In a `for` loop, `guard_hits += 1` is obviously correct. Here, a node
returns a *partial* state which LangGraph merges, so accumulating a counter means reading the
old value and returning the new one — easy to get subtly wrong, and the reason the token
totals below are computed rather than incremented in place.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Everything the graph carries between nodes."""

    # `add_messages` is a *reducer*: a node returning {"messages": [x]} appends rather than
    # replaces. This is what keeps the history append-only, which is what keeps the prefix
    # byte-stable, which is what keeps the server's KV cache valid across turns. Every other
    # key in this TypedDict has no reducer, so returning it overwrites.
    messages: Annotated[list[AnyMessage], add_messages]

    # Which model turn we are on, 1-based. This is the step budget from the original loop:
    # the framework's own `recursion_limit` counts node executions, which is a different and
    # less meaningful number — two nodes run per turn, plus a nudge sometimes.
    step: int

    # Cost is reported, not hidden. The total is the bill; the peak is how close a single
    # prompt came to the context window. They answer different questions.
    prompt_tokens: int
    completion_tokens: int
    peak_prompt_tokens: int

    # Loop-guard state: the previous call's identity, and how many times it has repeated.
    last_signature: str | None
    guard_hits: int


def initial_state(system: str, task: str) -> AgentState:
    """The only two messages that exist before the model has done anything."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return AgentState(
        messages=[SystemMessage(content=system), HumanMessage(content=task)],
        step=0,
        prompt_tokens=0,
        completion_tokens=0,
        peak_prompt_tokens=0,
        last_signature=None,
        guard_hits=0,
    )
