"""Observability: one record per model turn and per tool call.

Small, but the thing that makes the agent debuggable. An agent that prints only SOLVED or
NOT SOLVED gives you nothing to act on; a trace shows which turn went wrong, how much
context each turn cost, and where the time went. `agentfix solve --verbose` prints it live.

LangChain has its own callback system, and LangSmith is the production answer. This stays
hand-rolled for one reason: it is the only way to print a trace that lines up line-for-line
with the no-framework edition, which is how you check that the port changed the code and
not the agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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


class Tracer:
    """Collects events, and optionally prints them as they happen."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.events: list[TraceEvent] = []

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
