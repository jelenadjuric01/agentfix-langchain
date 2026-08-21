"""agentfix — a coding agent small enough to read end to end, built on LangGraph.

An agent here is a bounded graph that asks a model what to do, does it, and verifies the
result by running the tests rather than by believing the model.

This is the *framework* edition. The no-framework original is a sibling repository, and the
interesting parts of this one are the places where the two differ. Suggested reading order:

  1. tools/base.py       what a tool is, and the limits on what it may return
  2. tasks/loader.py     what a task is; the copy-to-tempdir context manager
  3. tools/fs.py         list_files, read_file, write_file
  4. tools/tests_tool.py run_tests — the agent's only oracle
  5. agent/graph.py      the agent. If you read one file, read this one.
  6. runner.py           how the pieces above are wired together

Then, as needed: llm/ (the real and scripted models), sandbox/ (how tests are executed),
eval/ (measurement), doctor.py and cli.py (entry points).

ARCHITECTURE.md annotates the same graph with the design decisions and their measurements.
"""

__version__ = "0.1.0"
