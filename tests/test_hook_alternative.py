"""What it would cost to build the loop guard on the framework's own hook instead.

`agent/graph.py` guards inside `tools_node` and hands the surviving calls to `ToolNode` in one
batched invocation. LangGraph offers a different shape for the same job: `create_react_agent`
takes a `post_model_hook`, and the router behind it (langgraph/prebuilt/chat_agent_executor.py)
computes

    pending = [c for c in last_ai_message.tool_calls if c["id"] not in answered_tool_call_ids]

and `Send`s one task per pending call. Answer a call in the hook and it is simply not pending,
so it never runs — the guard contract, supplied by the framework.

That shape is reproducible in a hand-built graph: `Send` is public and the diff is four lines.
So the question is not whether it is possible but what it costs, and these two tests are the
answer. Neither asserts anything about OUR code; both pin down LangGraph behaviour that the
choice depends on, which is why they live here rather than in a comment that could rot.

Run:  uv run python -m unittest tests.test_hook_alternative -v
"""

from __future__ import annotations

import operator
import threading
import time
import unittest
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

SLEEP = 0.05


class FanState(TypedDict):
    calls: list[str]
    log: Annotated[list[str], operator.add]


class VerdictState(TypedDict):
    calls: list[str]
    log: Annotated[list[str], operator.add]
    # Deliberately WITHOUT a reducer, exactly as AgentState.tests_passed is. See state.py for
    # why: a reducer is handed (current, incoming) and cannot tell "the suite went green" from
    # "this call changed the workspace, so the verdict is void".
    tests_passed: bool


def _fan(state: Any) -> Any:
    return [Send("work", {"name": name}) for name in state["calls"]]


class TestSendFanOutRespectsMaxConcurrency(unittest.TestCase):
    """The oracle guarantee survives the hook's shape. This one is good news.

    `tools_node` pins `max_concurrency=1` on its single batched invocation because `ToolNode`
    otherwise runs a turn's calls in a real thread pool — which can let `run_tests` measure the
    workspace as it was before a `write_file` in the same turn, and that is a false SOLVED.

    The obvious worry about the hook's shape is that one task per call puts the calls in
    different supersteps' tasks, outside the reach of that one config. It does not:
    `max_concurrency` throttles the fan-out too.
    """

    def _run(self, config: dict[str, Any]) -> tuple[int, list[str]]:
        live = 0
        peak = 0
        lock = threading.Lock()

        def work(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(SLEEP)
            with lock:
                live -= 1
            return {"log": [payload["name"]]}

        graph = StateGraph(FanState)
        graph.add_node("start", lambda state: {})
        graph.add_node("work", work)
        graph.add_edge(START, "start")
        graph.add_conditional_edges("start", _fan, ["work"])
        graph.add_edge("work", END)
        app = graph.compile()

        result = app.invoke({"calls": ["a", "b", "c"], "log": []}, config=config)
        return peak, result["log"]

    def test_fan_out_runs_in_parallel_by_default(self):
        peak, log = self._run({})
        self.assertEqual(peak, 3, "three Sends ran concurrently — the default, and the hazard")
        self.assertEqual(log, ["a", "b", "c"], "message order is preserved either way")

    def test_max_concurrency_serialises_the_fan_out(self):
        peak, log = self._run({"max_concurrency": 1})
        self.assertEqual(peak, 1, "one at a time, so the oracle guarantee holds under Send too")
        self.assertEqual(log, ["a", "b", "c"])


class TestFanOutBreaksAReducerFreeKey(unittest.TestCase):
    """And this is the bill. One task per call makes the tool step a CONCURRENT writer.

    `tools_node` folds the verdict itself, in the same return as the tool replies, because it is
    the single writer of `tests_passed`. Fan the calls out and each task wants to write that key
    in one superstep, which LangGraph refuses for any key without a reducer.

    So adopting the hook's shape is not free: the verdict fold has to move to a node after the
    tools, or `tests_passed` has to take a reducer — and state.py argues against the reducer on
    its own merits. That is the trade, and it is the same cost as the synthetic AIMessage in
    `tools_node` today, paid in the state schema instead of in a comment.
    """

    def _run(self, calls: list[str]) -> dict[str, Any]:
        def work(payload: dict[str, Any]) -> dict[str, Any]:
            # As tools_node does today: reply, and fold the verdict in the same update.
            return {"log": [payload["name"]], "tests_passed": True}

        graph = StateGraph(VerdictState)
        graph.add_node("start", lambda state: {})
        graph.add_node("work", work)
        graph.add_edge(START, "start")
        graph.add_conditional_edges("start", _fan, ["work"])
        graph.add_edge("work", END)
        app = graph.compile()

        return app.invoke(
            {"calls": calls, "log": [], "tests_passed": False},
            config={"max_concurrency": 1},
        )

    def test_one_call_per_turn_is_fine(self):
        result = self._run(["a"])
        self.assertTrue(result["tests_passed"], "a single writer, so nothing to reconcile")

    def test_two_calls_in_one_turn_cannot_both_write_the_verdict(self):
        from langgraph.errors import InvalidUpdateError

        with self.assertRaises(InvalidUpdateError) as ctx:
            self._run(["a", "b"])
        self.assertIn("tests_passed", str(ctx.exception))
        self.assertIn("only one value per step", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
