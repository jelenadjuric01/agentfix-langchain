"""run_tests: the agent's only oracle."""

from __future__ import annotations

from agentfix.sandbox.base import ExecResult
from agentfix.tools.tests_tool import RunTestsTool
from tests.support import PYTHON_UNITTEST, FakeBackend, TempDirTestCase


class TestRunTestsTool(TempDirTestCase):
    def _tool(self, backend: FakeBackend) -> RunTestsTool:
        return RunTestsTool(root=self.tmp, command=PYTHON_UNITTEST, backend=backend)

    def test_declares_a_schema_the_model_can_use(self):
        tool = self._tool(FakeBackend())
        self.assertEqual(tool.name, "run_tests")
        self.assertTrue(tool.description, "the model chooses tools by their description")
        self.assertEqual(tool.args_schema.model_json_schema()["type"], "object")

    def test_a_failing_suite_is_a_normal_observation_not_a_tool_error(self):
        """Failing tests are the information the agent needs, not a malfunction."""
        out = self._tool(FakeBackend(ExecResult(False, "1 failed", 0.1))).invoke({})
        self.assertIn("Tests failed.", out)
        self.assertIn("1 failed", out)

    def test_the_verdict_is_the_first_line(self):
        """unittest buries its verdict at the end; a 12B model reads the top of the output."""
        out = self._tool(FakeBackend(ExecResult(True, "Ran 2 tests\nOK", 0.1))).invoke({})
        self.assertTrue(out.startswith("All tests passed."))

    def test_the_result_is_remembered_for_is_done(self):
        tool = self._tool(FakeBackend(ExecResult(True, "OK", 0.1)))
        self.assertIsNone(tool.last_result)
        tool.invoke({})
        self.assertIsNotNone(tool.last_result)
        self.assertTrue(tool.last_result.passed)

    def test_invalidate_discards_a_stale_green_result(self):
        tool = self._tool(FakeBackend(ExecResult(True, "OK", 0.1)))
        tool.invoke({})
        tool.invalidate()
        self.assertIsNone(tool.last_result, "a write must not leave a green result behind")

    def test_the_backend_is_asked_to_run_the_task_command_in_the_workspace(self):
        backend = FakeBackend()
        self._tool(backend).invoke({})
        workspace, command, _ = backend.calls[0]
        self.assertEqual(workspace, self.tmp)
        self.assertEqual(command, PYTHON_UNITTEST)
