"""Stage 2 — catching a model that has stopped learning.

The real graph, the real tools, a real temp directory, a scripted model. No Ollama needed.
"""

from __future__ import annotations

import sys
import unittest

from agentfix.agent.graph import MAX_GUARD_HITS, call_signature, run_agent
from agentfix.agent.trace import Tracer
from agentfix.llm.fake import (
    FakeChatModel,
    assistant_invalid_tool_call,
    assistant_text,
    assistant_tool_call,
)
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool
from tests.support import TempDirTestCase, make_task

BUGGY = "def total(prices):\n    return sum(prices) - 1\n"
SUITE = (
    "import unittest\n\n"
    "from cart import total\n\n\n"
    "class TestCart(unittest.TestCase):\n"
    "    def test_total(self):\n"
    "        self.assertEqual(total([1, 2]), 3)\n"
)


class TestCallSignature(unittest.TestCase):
    """The identity function. One line, and one trap."""

    def test_the_same_call_twice_has_the_same_signature(self):
        call = {"name": "read_file", "args": {"path": "cart.py"}, "id": "c1"}
        other = {"name": "read_file", "args": {"path": "cart.py"}, "id": "c2"}
        self.assertEqual(call_signature(call), call_signature(other))
        self.assertNotIn("c1", call_signature(call), "the id must not be part of the identity")

    def test_different_arguments_are_a_different_call(self):
        a = {"name": "read_file", "args": {"path": "cart.py"}, "id": "c1"}
        b = {"name": "read_file", "args": {"path": "other.py"}, "id": "c2"}
        self.assertNotEqual(call_signature(a), call_signature(b))

    def test_a_different_tool_is_a_different_call(self):
        a = {"name": "read_file", "args": {}, "id": "c1"}
        b = {"name": "list_files", "args": {}, "id": "c2"}
        self.assertNotEqual(call_signature(a), call_signature(b))

    def test_key_order_does_not_change_the_signature(self):
        """The trap. JSON object key order is not meaningful, and a model will reorder keys.

        A signature built by stringifying the dict as-is lets a stuck model walk straight past
        the guard by shuffling its arguments.
        """
        a = {"name": "write_file", "args": {"path": "a.py", "content": "x = 1\n"}, "id": "c1"}
        b = {"name": "write_file", "args": {"content": "x = 1\n", "path": "a.py"}, "id": "c2"}
        self.assertEqual(call_signature(a), call_signature(b))

    def test_a_call_with_no_arguments_works(self):
        """run_tests and list_files take none, so this must not raise."""
        self.assertTrue(call_signature({"name": "run_tests", "args": {}, "id": "c1"}))


class Stage2TestCase(TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text(BUGGY, encoding="utf-8")
        (self.tmp / "test_cart.py").write_text(SUITE, encoding="utf-8")
        self.task = make_task(self.tmp)
        self.tools = [
            ListFilesTool(root=self.tmp),
            ReadFileTool(root=self.tmp),
            WriteFileTool(root=self.tmp),
            RunTestsTool(
                root=self.tmp,
                command=(sys.executable, "-m", "unittest", "discover", "-q"),
                backend=SubprocessBackend(),
                timeout_s=30,
            ),
        ]

    def run_with(self, replies, max_steps=None, tracer=None):
        replies = list(replies)
        llm = FakeChatModel(replies=replies)
        result = run_agent(
            self.task,
            llm,
            self.tools,
            max_steps=len(replies) if max_steps is None else max_steps,
            tracer=tracer,
        )
        return result, llm


class TestTheGuard(Stage2TestCase):
    def test_an_identical_repeated_call_is_not_executed_again(self):
        """Re-running it would spend a step to learn nothing. Say so instead."""
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
                assistant_text("done"),
            ],
            max_steps=3,
            tracer=tracer,
        )
        guarded = [e for e in tracer.events if "guarded" in e.detail]
        self.assertEqual(len(guarded), 1, "the second identical call must be refused, not run")

    def test_the_refused_call_still_gets_an_answer(self):
        """The rule you must not break: one reply per call, including the refusals.

        Skip the `tool_call_id` and the model's next request is rejected by the API — a failure
        that surfaces one turn away from the code that caused it.
        """
        _, llm = self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="first"),
                assistant_tool_call("list_files", {}, call_id="second"),
                assistant_text("done"),
            ],
            max_steps=3,
        )
        answered = [m.tool_call_id for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertEqual(answered, ["first", "second"])

    def test_the_model_is_told_why_nothing_happened(self):
        _, llm = self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
                assistant_text("done"),
            ],
            max_steps=3,
        )
        answers = [str(m.content) for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertTrue(
            any("already called" in a for a in answers),
            "an unexplained refusal teaches the model nothing",
        )

    def test_key_order_does_not_defeat_the_guard_end_to_end(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call(
                    "write_file", {"path": "a.py", "content": "x = 1\n"}, call_id="c1"
                ),
                assistant_tool_call(
                    "write_file", {"content": "x = 1\n", "path": "a.py"}, call_id="c2"
                ),
                assistant_text("done"),
            ],
            max_steps=3,
            tracer=tracer,
        )
        self.assertTrue(any("guarded" in e.detail for e in tracer.events))

    def test_making_progress_resets_the_counter(self):
        """A → B → A is not a stuck model. Only consecutive repeats are."""
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("run_tests", {}, call_id="c2"),
                assistant_tool_call("list_files", {}, call_id="c3"),
                assistant_text("done"),
            ],
            max_steps=4,
            tracer=tracer,
        )
        self.assertFalse(
            any("guarded" in e.detail for e in tracer.events),
            "nothing here repeats consecutively",
        )

    def test_a_stuck_model_is_abandoned_rather_than_burning_the_budget(self):
        """The payoff. Ten identical calls must not cost ten model turns."""
        result, llm = self.run_with(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(10)],
            max_steps=10,
        )
        self.assertFalse(result.solved)
        self.assertLessEqual(
            llm.index, MAX_GUARD_HITS + 1, "the run should have been given up on, not run out"
        )

    def test_repeated_malformed_json_also_trips_the_guard(self):
        """A model stuck on one broken call is just as stuck as one stuck on a good call."""
        result, llm = self.run_with(
            [
                assistant_invalid_tool_call("read_file", '{"path": ', call_id=f"c{i}")
                for i in range(10)
            ],
            max_steps=10,
        )
        self.assertFalse(result.solved)
        self.assertLessEqual(llm.index, MAX_GUARD_HITS + 1)


class TestTheAgentStillWorks(Stage2TestCase):
    def test_a_normal_run_is_untouched_by_the_guard(self):
        """The guard must be invisible to an agent that is making progress."""
        fixed = "def total(prices):\n    return sum(prices)\n"
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("read_file", {"path": "cart.py"}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": fixed}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed it."),
            ]
        )
        self.assertTrue(result.solved)


if __name__ == "__main__":
    unittest.main()
