"""The trace: what makes a run debuggable rather than just pass/fail."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from agentfix.agent.trace import DETAIL_CLIP, Tracer, TraceEvent


class TestTracer(unittest.TestCase):
    def test_events_are_collected_in_order(self):
        tracer = Tracer()
        tracer.record(TraceEvent(1, "llm", "assistant", "calls run_tests", 120, 0.4))
        tracer.record(TraceEvent(1, "tool", "run_tests", "Tests failed.", 120, 0.3))
        self.assertEqual([e.kind for e in tracer.events], ["llm", "tool"])

    def test_quiet_by_default(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer().record(TraceEvent(1, "llm", "assistant", "x", 1, 0.1))
        self.assertEqual(buffer.getvalue(), "")

    def test_verbose_prints_one_line_per_event(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer(verbose=True).record(
                TraceEvent(2, "tool", "run_tests", "Tests failed.", 300, 1.2)
            )
        printed = buffer.getvalue()
        self.assertIn("step 2", printed)
        self.assertIn("run_tests", printed)
        self.assertIn("300 tok", printed)
        self.assertEqual(len(printed.strip().splitlines()), 1)

    def test_newlines_are_flattened_and_long_detail_is_clipped(self):
        """One event, one line — a readable trace beats a faithful dump of file contents."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer(verbose=True).record(
                TraceEvent(1, "tool", "read_file", "a\nb\n" + "x" * 5000, 10, 0.0)
            )
        printed = buffer.getvalue()
        self.assertEqual(len(printed.strip().splitlines()), 1)
        self.assertLess(len(printed), DETAIL_CLIP + 200)

    def test_as_json_is_serialisable_for_the_eval_report(self):
        tracer = Tracer()
        tracer.record(TraceEvent(1, "llm", "assistant", "x", 120, 0.4))
        self.assertEqual(tracer.as_json()[0]["prompt_tokens"], 120)
        self.assertEqual(tracer.as_json()[0]["step"], 1)
