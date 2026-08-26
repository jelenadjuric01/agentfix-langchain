"""A scripted stand-in for the model, so the graph can be tested with nothing running.

This is why the test suite is fast, offline and deterministic: instead of mocking the graph's
internals, you hand it a *list of replies* and let the real graph execute against real tools in
a real temp directory. Nothing is patched — `FakeChatModel` is a real `BaseChatModel`, so the
graph cannot tell the difference.

A test reads like a screenplay of a model's turns:

    llm = FakeChatModel(replies=[
        assistant_tool_call("run_tests", {}),
        assistant_tool_call("read_file", {"path": "shopcart/cart.py"}),
        assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": FIXED}),
        assistant_tool_call("run_tests", {}),
        assistant_text("Fixed the tax rounding."),
    ])

The tests really are red before that write and really are green after it, because the fake
replaces only the model — never the tools, the sandbox, or the graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def assistant_text(text: str, prompt_tokens: int = 10) -> AIMessage:
    """A prose reply with no tool calls — the model talking rather than acting."""
    # The completion count is a stand-in. Nothing asserts on the value; it just has to be
    # plausible so the accounting in the graph has something to add up.
    return AIMessage(content=text, usage_metadata=_usage(prompt_tokens, len(text.split())))


def assistant_tool_call(
    name: str, arguments: dict[str, Any], call_id: str = "call_1", prompt_tokens: int = 10
) -> AIMessage:
    """A reply requesting one tool call."""
    return assistant_tool_calls(
        [(name, arguments)], call_ids=(call_id,), prompt_tokens=prompt_tokens
    )


def assistant_tool_calls(
    calls: Sequence[tuple[str, dict[str, Any]]],
    call_ids: Sequence[str] | None = None,
    prompt_tokens: int = 10,
    text: str = "",
) -> AIMessage:
    """A reply requesting SEVERAL tool calls in one turn, which a real model may do.

    The API permits any number of calls per assistant message, and requires an answer to every
    one of them: skip a `tool_call_id` and the next request is rejected. The graph therefore
    iterates over the calls — and without this builder there was no way to exercise that
    iteration with more than one element, so a graph answering only the first call would have
    passed every test in the suite.

        assistant_tool_calls([("run_tests", {}), ("list_files", {})], call_ids=("c1", "c2"))

    `call_ids` defaults to call_1..call_N. Ids must be distinct, since the whole point of an id
    is to tell two calls apart.
    """
    if call_ids is None:
        call_ids = [f"call_{index}" for index in range(1, len(calls) + 1)]
    ids = tuple(call_ids)
    assert len(ids) == len(calls), "one call id per call"
    assert len(set(ids)) == len(ids), "call ids must be distinct"

    return AIMessage(
        content=text,
        tool_calls=[
            {"name": name, "args": arguments, "id": call_id}
            for call_id, (name, arguments) in zip(ids, calls, strict=True)
        ],
        usage_metadata=_usage(prompt_tokens, 5 * len(calls)),
    )


class FakeChatModel(BaseChatModel):
    """Scripted client so the agent graph is testable with no model running."""

    replies: list[AIMessage]
    index: int = 0
    # Every history this model was called with, for tests that assert on what the graph
    # actually sent — that it is append-only, that the tool_call_id came back, and so on.
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-agentfix"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        """Ignore the tools on purpose: the replies are scripted, so nothing inspects schemas.

        Overridden because `BaseChatModel.bind_tools` raises NotImplementedError, and returning
        `self` (rather than a `RunnableBinding`) keeps `index` and `calls` observable from the
        test that built this object.
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # An assert, not an IndexError, because the message is the diagnosis: if the graph asks
        # for more turns than the test scripted, the test's model of the graph is wrong. This
        # fires whenever a change makes the agent take an extra step.
        assert self.index < len(self.replies), (
            f"FakeChatModel script exhausted after {self.index} call(s); "
            "the agent asked for more turns than the test scripted"
        )
        # Snapshot the history at this turn, since the graph keeps appending.
        self.calls.append(list(messages))
        reply = self.replies[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])
