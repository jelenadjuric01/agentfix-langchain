"""Stage 1 — where a run can end, and on whose word.

Every test here drives the REAL graph against the REAL tools in a real temp directory. Only the
model is scripted, so none of this needs Ollama running.
"""

from __future__ import annotations

import sys
import unittest

from agentfix.agent.graph import NUDGE, run_agent
from agentfix.llm.fake import (
    FakeChatModel,
    assistant_text,
    assistant_tool_call,
    assistant_tool_calls,
)
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool
from tests.support import TempDirTestCase, make_task

BUGGY = "def total(prices):\n    return sum(prices) - 1\n"
FIXED = "def total(prices):\n    return sum(prices)\n"
SUITE = (
    "import unittest\n\n"
    "from cart import total\n\n\n"
    "class TestCart(unittest.TestCase):\n"
    "    def test_total(self):\n"
    "        self.assertEqual(total([1, 2]), 3)\n"
)


class Stage1TestCase(TempDirTestCase):
    """A real one-file project that starts red, plus the four tools bound to it."""

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


class TestToolCallsAlwaysRun(Stage1TestCase):
    def test_a_turn_that_asks_for_a_tool_goes_to_the_tools(self):
        """Rule 1. Without it the agent can never do anything at all."""
        result, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("read_file", {"path": "cart.py"}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the off-by-one."),
            ]
        )
        self.assertTrue(result.solved)
        self.assertEqual((self.tmp / "cart.py").read_text(), FIXED)
        self.assertEqual(llm.index, 5, "the graph took exactly the scripted turns")

    def test_the_run_ends_when_the_tests_pass(self):
        """Rule 2."""
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertTrue(result.solved)

    def test_a_model_that_merely_claims_success_is_not_believed(self):
        """The whole point of the design. The tests were never run; nothing is green."""
        result, _ = self.run_with(
            [
                assistant_text("I have fixed the bug."),
                assistant_text("Really, it is fixed."),
            ],
            max_steps=2,
        )
        self.assertFalse(result.solved, "prose is not evidence")

    def test_the_agent_cannot_end_successfully_on_a_red_suite(self):
        """It ran the tests, they failed, and then it gave up talking."""
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("This looks unfixable to me."),
            ],
            max_steps=2,
        )
        self.assertFalse(result.solved)


class TestTheNudge(Stage1TestCase):
    def test_prose_on_a_red_suite_sends_the_model_back(self):
        """Rule 3. A text-only reply is not a stop condition, so it costs a nudge and a turn."""
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("I think it is fine now."),
                assistant_text("still fine"),
            ],
            max_steps=3,
        )
        self.assertTrue(
            any(getattr(m, "content", None) == NUDGE for m in llm.calls[-1]),
            "the model should have been told the tests are still failing",
        )

    def test_the_budget_still_ends_the_run(self):
        """Rule 3's exception: no nudging past the step budget, or a stubborn model never stops."""
        result, llm = self.run_with(
            [assistant_text(f"still fine {i}") for i in range(10)], max_steps=3
        )
        self.assertFalse(result.solved)
        self.assertEqual(llm.index, 3, "the run must not exceed its step budget")


class TestItIsStillAnAgent(Stage1TestCase):
    def test_several_calls_in_one_turn_are_all_answered(self):
        """A model may ask for more than one thing at a time, and the API requires every
        `tool_call_id` to come back answered."""
        _, llm = self.run_with(
            [
                assistant_tool_calls(
                    [("run_tests", {}), ("list_files", {})], call_ids=("c1", "c2")
                ),
                assistant_text("done"),
            ]
        )
        ids = [m.tool_call_id for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertEqual(ids, ["c1", "c2"])

    def test_a_write_after_a_green_run_is_not_still_solved(self):
        """The verdict is about the code as it stands, not the best it ever looked."""
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": BUGGY}),
                assistant_text("done"),
            ]
        )
        self.assertFalse(result.solved)


if __name__ == "__main__":
    unittest.main()
