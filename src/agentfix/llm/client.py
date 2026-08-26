"""The real model client: a `ChatOllama` pointed at the local Ollama server.

The only file in the project that performs network I/O. It is one function, because the
framework's job is exactly this: the client speaks Ollama's protocol, parses tool calls into
`AIMessage.tool_calls` and records token usage in `usage_metadata`. The no-framework edition
hand-wrote all of that. What it does NOT do is rescue unparseable tool arguments: those raise
out of the model call.

`ChatOllama` rather than `ChatOpenAI`, and the history is worth knowing because it is the
sharpest lesson in the project about using a framework's *right* integration.

Ollama serves an OpenAI-compatible endpoint at `/v1`, so `ChatOpenAI` works against it — and
this repo used it, to keep the wire format byte-identical to the no-framework original for
comparison. It cost two silent misconfigurations, both of them measured:

  - `ChatOpenAI.max_tokens` is aliased to OpenAI's newer `max_completion_tokens`, and that is
    the key it puts on the wire. Ollama's `/v1` ignores it. Asking for 8 tokens:
    `max_completion_tokens=8` produced 692; `max_tokens=8` produced 8. The cap has to reach
    the server, because it is the ceiling on how large a file `write_file` can emit in one
    turn — so the value had to be smuggled through `extra_body`.
  - `/v1` drops the `options` block entirely, so `num_ctx` never arrived either (`ollama ps`
    still reported 4096) and the context window had to be baked into a derived model instead.
    See Modelfile.

`ChatOllama` talks to Ollama's own API, where both are first-class typed fields. Measured on
this machine, against the same server:

    num_predict=8               ->   8 tokens generated  (uncapped: 891)
    num_ctx=8192 on a model     -> `ollama ps` reports 8192, overriding the Modelfile's 16384
    derived with num_ctx 16384

So the cap is real and the context window is now negotiated per request. The Modelfile still
works and is still what the README tells you to build — it is belt and braces now rather than
the only way in.

The general lesson, and the reason this file is worth reading twice: a compatibility endpoint
accepts the requests it does not honour. Nothing errored, nothing warned, and two of the three
settings that decide whether this agent works were being discarded in transit.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from agentfix.config import LLMConfig


def make_chat_model(config: LLMConfig | None = None) -> ChatOllama:
    """Build the model client. No api_key: Ollama has no authentication to satisfy."""
    config = config or LLMConfig.from_env()
    return ChatOllama(
        base_url=config.api_url,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        # Ollama's names for the two caps. `num_predict` is the ceiling on ONE reply, and
        # `num_ctx` is the context window the model is loaded with — both honoured here, which
        # is the entire reason this file uses ChatOllama. See the module docstring.
        num_predict=config.max_tokens,
        num_ctx=config.num_ctx,
    )
