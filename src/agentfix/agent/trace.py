"""Observability: one record per model turn and per tool call.

Small, but the thing that makes the agent debuggable. An agent that prints only SOLVED or
NOT SOLVED gives you nothing to act on; a trace shows which turn went wrong, how much
context each turn cost, and where the time went. `agentfix solve --verbose` prints it live.

This is a `BaseCallbackHandler`, which is the framework's own observability seam: it is handed
to the graph once, as `config={"callbacks": [tracer]}`, and LangChain then calls the hooks
below as the model and the tools run. The graph's nodes contain no tracing code at all — they
are state transitions and nothing else, which is the whole reason to prefer this over an
observer the nodes have to remember to call.

Two things the hooks cannot see, because they never happened: a call the loop guard refused to
execute, and a tool call whose JSON never parsed. No tool ran, so no tool callback fires. The
graph reports those itself with `note`, and that split is the honest one — the framework
observes what the framework did.

LangSmith is the production answer to all of this and speaks the same interface; it would slot
in beside this handler rather than replacing it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

# Long enough to read a sentence of model reasoning, short enough that one event stays on
# one line. A readable 20-line trace beats a faithful 500-line dump of file contents.
DETAIL_CLIP = 250


@dataclass(frozen=True)
class TraceEvent:
    """One thing that happened, at a known step. Frozen: history is recorded, not rewritten."""

    step: int  # which graph iteration, 1-based
    kind: str  # "llm" or "tool"
    name: str  # "assistant", or the tool's name
    detail: str  # the model's message, or the tool's output
    prompt_tokens: int  # context size at this point — watch it grow across a run
    latency_s: float


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


def prompt_tokens_of(message: AIMessage) -> int:
    """The context size the server reported for this turn, or 0 if it reported none."""
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    return int(usage.get("input_tokens", 0))


class Tracer(BaseCallbackHandler):
    """Collects events from LangChain's callbacks, and optionally prints them as they happen."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.events: list[TraceEvent] = []

        # Which model turn we are in. Counted here rather than read from the graph's state
        # because a callback is handed no state — and the number is the same one, since a turn
        # begins precisely when the model is called.
        self.step = 0

        # The context size of the current turn, so a tool line reports the same `ctx` as the
        # model line that asked for it. Remembered rather than passed around: tool callbacks
        # know nothing about the message that requested the call.
        self.turn_prompt_tokens = 0

        # run_id -> (start time, name). Keyed by run because callbacks are not guaranteed to
        # nest, and a tool that fails never reaches on_tool_end.
        #
        # The name is stored because `on_tool_error` is NOT told which tool raised: LangChain
        # passes it only run_id, parent_run_id, tags and tool_call_id. `on_tool_start` is the
        # one hook that gets the name, in `serialized`. Without stashing it here, the single
        # trace line you most need to identify reads "tool:tool".
        self._started: dict[UUID, tuple[float, str]] = {}

    # --- what the framework observes --------------------------------------------------------

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        """A model turn is starting. This is the only place `step` advances."""
        self.step += 1
        self._mark(kwargs.get("run_id"), "assistant")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # Indexing defensively, and recording when there is nothing to record. A raise in here
        # would be swallowed: `BaseCallbackHandler.raise_error` is False, so LangChain logs a
        # WARNING on a logger nothing in this project configures and carries on — the run
        # survives and the event silently disappears from the trace.
        generations = response.generations
        reply = generations[0][0] if generations and generations[0] else None
        message = getattr(reply, "message", None)
        if not isinstance(message, AIMessage):
            # A turn that produced no assistant message still happened, and `step` has already
            # advanced. Say so, and clear the token count rather than letting this turn's tool
            # lines inherit the PREVIOUS turn's context size — a trace that is quietly wrong
            # about context growth is worse than one with a gap in it.
            self.turn_prompt_tokens = 0
            self.note("llm", "assistant", "no assistant message in the model's reply")
            return
        reply_message = message
        self.turn_prompt_tokens = prompt_tokens_of(reply_message)
        self.record(
            TraceEvent(
                step=self.step,
                kind="llm",
                name="assistant",
                detail=describe(reply_message),
                prompt_tokens=self.turn_prompt_tokens,
                latency_s=self._elapsed(kwargs.get("run_id")),
            )
        )

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """The only hook told which tool this is — `serialized["name"]`. Remember it."""
        self._mark(kwargs.get("run_id"), str(serialized.get("name") or "tool"))

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        remembered = self._name(kwargs.get("run_id"))
        name = output.name if isinstance(output, ToolMessage) else remembered
        detail = str(output.content) if isinstance(output, ToolMessage) else str(output)
        self.record(
            TraceEvent(
                step=self.step,
                kind="tool",
                name=str(name or "tool"),
                detail=detail,
                prompt_tokens=self.turn_prompt_tokens,
                latency_s=self._elapsed(kwargs.get("run_id")),
            )
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """A tool raised. ToolNode turns this into an observation; the trace should say so."""
        run_id = kwargs.get("run_id")
        # Read the name BEFORE `_elapsed`, which pops the entry. Sequenced explicitly rather
        # than relying on argument evaluation order.
        name = self._name(run_id)
        self.record(
            TraceEvent(
                step=self.step,
                kind="tool",
                name=name,
                detail=f"raised {type(error).__name__}: {error}",
                prompt_tokens=self.turn_prompt_tokens,
                latency_s=self._elapsed(run_id),
            )
        )

    # --- what only the graph knows ----------------------------------------------------------

    def note(self, kind: str, name: str, detail: str) -> None:
        """Record something the graph decided rather than something the framework ran.

        Two callers, both cases where no tool executed and so no callback fired: a call the
        loop guard refused, and a tool call whose arguments were not valid JSON. Latency is
        0.0 because nothing happened — that is the point.
        """
        self.record(TraceEvent(self.step, kind, name, detail, self.turn_prompt_tokens, 0.0))

    # --- recording ---------------------------------------------------------------------------

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.verbose:
            # → "we asked the model", ← "something came back".
            marker = "→" if event.kind == "llm" else "←"
            detail = event.detail.replace("\n", " ")[:DETAIL_CLIP]
            print(
                f"  step {event.step} {marker} {event.kind}:{event.name}  "
                f"[ctx {event.prompt_tokens} tok, {event.latency_s:.1f}s]  {detail}"
            )

    def as_json(self) -> list[dict[str, Any]]:
        """Serialise for the eval report. `asdict` turns each dataclass into a plain dict."""
        return [asdict(event) for event in self.events]

    # --- timing ------------------------------------------------------------------------------

    def _mark(self, run_id: UUID | None, name: str) -> None:
        if run_id is not None:
            self._started[run_id] = (time.time(), name)

    def _name(self, run_id: UUID | None) -> str:
        """What the start callback said this run was, without consuming the entry."""
        entry = self._started.get(run_id) if run_id is not None else None
        return "tool" if entry is None else entry[1]

    def _elapsed(self, run_id: UUID | None) -> float:
        """Seconds since the matching start callback. `pop` so the map cannot grow forever."""
        entry = self._started.pop(run_id, None) if run_id is not None else None
        return 0.0 if entry is None else round(time.time() - entry[0], 2)
