"""The agent graph: the step budget, the verification stop condition, and the loop guard.

These tests run the REAL graph against REAL tools in a REAL temp directory. Only the model is
replaced. Nothing is patched — that is what makes them meaningful: the suite really is red
before the agent's write and really is green after it.
"""

from __future__ import annotations

import sys

from agentfix.agent.graph import MAX_GUARD_HITS, NUDGE, is_done, run_agent, system_prompt
from agentfix.agent.trace import Tracer
from agentfix.llm.fake import (
    FakeChatModel,
    assistant_invalid_tool_call,
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


class GraphTestCase(TempDirTestCase):
    """A real one-file project that starts red, plus the tools bound to it."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text(BUGGY, encoding="utf-8")
        (self.tmp / "test_cart.py").write_text(SUITE, encoding="utf-8")
        self.task = make_task(self.tmp)
        self.run_tests = RunTestsTool(
            root=self.tmp,
            command=(sys.executable, "-m", "unittest", "discover", "-q"),
            backend=SubprocessBackend(),
            timeout_s=30,
        )
        self.tools = [
            ListFilesTool(root=self.tmp),
            ReadFileTool(root=self.tmp),
            WriteFileTool(root=self.tmp, on_write=self.run_tests.invalidate),
            self.run_tests,
        ]

    def run_with(self, replies, max_steps=None, tracer=None):
        # The script is the budget by default. A prose reply while the tests are still red is
        # nudged rather than accepted, and the nudge costs a turn — so a script of N replies
        # needs exactly N steps, or the fake runs off the end of its own screenplay.
        replies = list(replies)
        max_steps = len(replies) if max_steps is None else max_steps
        llm = FakeChatModel(replies=replies)
        result = run_agent(
            self.task, llm, self.tools, self.run_tests, max_steps=max_steps, tracer=tracer
        )
        return result, llm


class TestSystemPrompt(GraphTestCase):
    def test_the_tool_names_come_from_the_registered_tools(self):
        """Derived, not hardcoded, so the prompt and the schemas cannot drift apart."""
        prompt = system_prompt(self.tools)
        for name in ("list_files", "read_file", "write_file", "run_tests"):
            self.assertIn(name, prompt)


class TestHappyPath(GraphTestCase):
    def test_the_agent_solves_the_task_and_reports_what_it_cost(self):
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
        self.assertEqual(result.steps_used, 5)
        self.assertEqual(llm.index, 5, "the graph took exactly the scripted turns")
        self.assertEqual((self.tmp / "cart.py").read_text(), FIXED)
        self.assertGreater(result.prompt_tokens, 0)
        self.assertGreater(result.peak_prompt_tokens, 0)

    def test_the_history_is_append_only(self):
        """Byte-stable prefix -> the server's KV cache stays valid across turns."""
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        for earlier, later in zip(llm.calls, llm.calls[1:]):
            self.assertEqual(later[: len(earlier)], earlier)

    def test_the_tool_call_id_is_carried_back(self):
        """The API pairs each answer to its question by id; skip one and the next call fails."""
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}, call_id="abc123"),
                assistant_text("done"),
            ]
        )
        answers = [m for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertEqual([m.tool_call_id for m in answers], ["abc123"])

    def test_several_calls_in_one_turn_are_all_answered(self):
        """A turn answering only the first call would otherwise pass every test."""
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


class TestFailuresBecomeObservations(GraphTestCase):
    """Every way a model can get a tool call wrong must be recoverable, not fatal."""

    def _last_tool_content(self, llm) -> str:
        tools = [m for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        return str(tools[-1].content)

    def test_a_hallucinated_tool_name_does_not_end_the_run(self):
        result, llm = self.run_with(
            [
                assistant_tool_call("frobnicate", {"x": 1}),
                assistant_text("giving up"),
            ]
        )
        self.assertFalse(result.solved)
        self.assertIn("frobnicate", self._last_tool_content(llm))

    def test_a_missing_required_argument_does_not_end_the_run(self):
        _, llm = self.run_with(
            [
                assistant_tool_call("read_file", {}),
                assistant_text("giving up"),
            ]
        )
        content = self._last_tool_content(llm).lower()
        self.assertTrue("path" in content or "required" in content)

    def test_arguments_that_are_not_valid_json_are_still_answered(self):
        """ToolNode ignores these entirely, producing no reply. The API requires one."""
        _, llm = self.run_with(
            [
                assistant_invalid_tool_call("read_file", '{"path": "cart.py"'),
                assistant_text("giving up"),
            ]
        )
        answers = [m for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertEqual(len(answers), 1, "an unanswered tool call breaks the next request")
        self.assertIn("JSON", str(answers[-1].content))

    def test_a_tool_that_raises_does_not_end_the_run(self):
        """ToolNode's default lets the exception through; the graph opts back in."""
        broken = ReadFileTool(root=self.tmp)

        def explode(**_kwargs):
            raise RuntimeError("disk on fire")

        object.__setattr__(broken, "_run", explode)
        self.tools[1] = broken
        result, llm = self.run_with(
            [
                assistant_tool_call("read_file", {"path": "cart.py"}),
                assistant_text("giving up"),
            ]
        )
        self.assertFalse(result.solved)
        self.assertTrue(self._last_tool_content(llm))


class TestStopCondition(GraphTestCase):
    def test_not_done_before_the_tests_have_ever_run(self):
        self.assertFalse(is_done(self.run_tests))

    def test_not_done_when_the_model_only_claims_success(self):
        """The whole point: a model announcing victory is not evidence."""
        result, _ = self.run_with(
            [
                assistant_text("I have fixed the bug."),
                assistant_text("Really, it is fixed."),
            ],
            max_steps=2,
        )
        self.assertFalse(result.solved)

    def test_done_only_once_the_tests_actually_pass(self):
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertTrue(result.solved)

    def test_a_prose_reply_while_red_is_nudged_rather_than_accepted(self):
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("I think it is fine now."),
                assistant_text("still fine"),
            ],
            max_steps=3,
        )
        self.assertTrue(any(getattr(m, "content", None) == NUDGE for m in llm.calls[-1]))

    def test_a_write_invalidates_a_previously_green_result(self):
        """Otherwise: run tests, pass, then break them, and is_done still says SOLVED."""
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": BUGGY}),
                assistant_text("done"),
            ]
        )
        self.assertFalse(result.solved, "a stale green result must not survive a write")

    def test_the_agent_cannot_fake_success_by_rewriting_the_tests(self):
        result, _ = self.run_with(
            [
                assistant_tool_call(
                    "write_file",
                    {"path": "test_cart.py", "content": "import unittest\n"},
                ),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertFalse(result.solved)
        self.assertIn("assertEqual", (self.tmp / "test_cart.py").read_text())


class TestBudgetAndGuard(GraphTestCase):
    def test_the_step_budget_is_respected(self):
        result, llm = self.run_with(
            [assistant_tool_call("list_files", {}) for _ in range(10)], max_steps=3
        )
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(llm.index, 3, "the graph must not exceed its budget")

    def test_an_identical_repeated_call_is_not_executed_again(self):
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
        self.assertEqual(len(guarded), 1)

    def test_key_order_does_not_defeat_the_guard(self):
        """{"a":1,"b":2} and {"b":2,"a":1} are the same call."""
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

    def test_a_stuck_model_is_abandoned_rather_than_looping_to_the_budget(self):
        result, llm = self.run_with(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(10)],
            max_steps=10,
        )
        self.assertFalse(result.solved)
        self.assertLessEqual(llm.index, MAX_GUARD_HITS + 1)

    def test_repeated_malformed_json_also_trips_the_guard(self):
        """Parity with the original, where bad arguments arrived as an ordinary call and were
        guarded for free. On this side they arrive on a separate field, so it takes code."""
        result, llm = self.run_with(
            [
                assistant_invalid_tool_call("read_file", '{"path": ', call_id=f"c{i}")
                for i in range(10)
            ],
            max_steps=10,
        )
        self.assertFalse(result.solved)
        self.assertLessEqual(llm.index, MAX_GUARD_HITS + 1, "a stuck model must be abandoned")

    def test_making_progress_resets_the_guard(self):
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
        self.assertFalse(any("guarded" in e.detail for e in tracer.events))


class TestTracing(GraphTestCase):
    def test_every_model_turn_and_tool_call_is_recorded(self):
        tracer = Tracer()
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        self.assertEqual([e.kind for e in tracer.events], ["llm", "tool", "llm"])
        self.assertEqual(result.trace, tuple(tracer.events))

    def test_a_tool_line_reports_the_context_size_of_the_turn_that_asked_for_it(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("run_tests", {}, prompt_tokens=1234),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        tool_event = next(e for e in tracer.events if e.kind == "tool")
        self.assertEqual(tool_event.prompt_tokens, 1234)

    def test_the_absence_of_reasoning_is_visible(self):
        """Measured on Mellum2: a tool-calling turn carries no reasoning text at all."""
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        self.assertIn("NO REASONING", tracer.events[0].detail)

    def test_reasoning_is_shown_when_the_model_does_emit_it(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_calls([("run_tests", {})], text="Let me see what fails first."),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        self.assertIn("Let me see what fails first.", tracer.events[0].detail)
