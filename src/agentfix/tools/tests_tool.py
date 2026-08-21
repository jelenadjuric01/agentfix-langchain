"""The run_tests tool — the agent's only oracle, and the only sandboxed tool.

This is the most consequential tool in the project. `is_done` in agent/graph.py consults
`last_result` and nothing else, so a run ends successfully only because the tests actually
passed — never because the model announced it was finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from langchain_core.tools import BaseTool

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import NoArgs


class RunTestsTool(BaseTool):
    """Runs the task's test command through an execution backend and remembers the result."""

    name: str = "run_tests"
    # The model is told outright that this is the source of truth. It is also *asked*, never
    # told: nothing hands the agent the failing test output up front, so discovering the
    # failure is part of the task.
    description: str = (
        "Run the project's test suite and return the result. This is the source of truth."
    )
    args_schema: type[BaseModel] = NoArgs

    root: Path
    command: tuple[str, ...]
    # Injected rather than constructed here, so tests can pass a fake backend and the
    # subprocess/docker choice stays the caller's (runner.py calls get_backend()).
    backend: Any
    timeout_s: int = 10

    # The one piece of mutable state in the tool layer, and the agent's whole verdict.
    last_result: ExecResult | None = None

    # `ExecutionBackend` is a Protocol and `ExecResult` a dataclass; neither is something
    # pydantic can validate, so they have to be allowed explicitly.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def execution_backend(self) -> ExecutionBackend:
        """`backend` typed as Any for pydantic's sake; read it back with the real type."""
        backend: ExecutionBackend = self.backend
        return backend

    def invalidate(self) -> None:
        """The workspace changed, so the last test run is no longer evidence about it.

        Wired up in runner.py as `WriteFileTool(root=..., on_write=run_tests.invalidate)`.
        Without it an agent could run the tests, see them pass, then write a file that breaks
        them — and `is_done` would still see the stale green result and report SOLVED.
        """
        self.last_result = None

    def _run(self) -> str:
        result = self.execution_backend.run(self.root, self.command, timeout_s=self.timeout_s)
        self.last_result = result

        # Note this returns a normal observation even when the tests fail: the *tool* worked.
        # Failing tests are the information the agent needs, not a tool error. The headline is
        # prepended because unittest buries the verdict at the end of its report, and a 12B
        # model reading 2,000 characters of traceback does better when the answer is first.
        headline = "All tests passed." if result.passed else "Tests failed."
        return f"{headline}\n\n{result.output}".strip()
