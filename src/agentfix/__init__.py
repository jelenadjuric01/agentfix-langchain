"""agentfix — a coding agent small enough to read end to end, built on LangGraph.

An agent here is a bounded graph that asks a model what to do, does it, and verifies the
result by running the tests rather than by believing the model.

This is the *framework* edition. The no-framework original is a sibling repository, and the
interesting parts of this one are the places where the two differ. Suggested reading order:

  1. tools/base.py       what a tool is, the limits on what it may return, and the artifact
                         channel it reports through
  2. tasks/loader.py     what a task is; the copy-to-tempdir context manager
  3. tools/fs.py         list_files, read_file, write_file
  4. tools/tests_tool.py run_tests — the agent's only oracle
  5. agent/state.py      what the graph carries, and the reducers that combine it
  6. agent/graph.py      the agent. If you read one file, read this one.
  7. runner.py           how the pieces above are wired together

Then, as needed: agent/trace.py (observability, as a LangChain callback handler), llm/ (the
real and scripted models), sandbox/ (how tests are executed), eval/ (measurement),
doctor.py and cli.py (entry points).

agent/prebuilt.py is the argument rather than the implementation: the same agent built from
`langchain.agents.create_agent` and its middleware, with the three invariants above expressed
as middleware where that is possible and documented where it is not. Optional dependency, not
on the path `agentfix solve` takes.
"""

__version__ = "0.1.0"
