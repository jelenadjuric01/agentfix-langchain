"""The real model client: a `ChatOpenAI` pointed at Ollama.

The only file in the project that performs network I/O. It is one function, because the
framework's job is exactly this: `ChatOpenAI` speaks the OpenAI protocol, parses tool calls
into `AIMessage.tool_calls`, records token usage in `usage_metadata`, and puts unparseable
arguments in `invalid_tool_calls`. The no-framework edition hand-wrote all of that.

`ChatOpenAI` rather than `ChatOllama` is a deliberate choice for this edition: it keeps the
wire format byte-identical to the original, so a trace from this repo can be compared
line-for-line against one from the no-framework repo. The cost is inherited below.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agentfix.config import LLMConfig


def make_chat_model(config: LLMConfig | None = None) -> ChatOpenAI:
    """Build the model client. Works against Ollama, vLLM, or any /v1 endpoint."""
    config = config or LLMConfig.from_env()
    return ChatOpenAI(
        base_url=config.base_url,
        # The api_key argument is required: the SDK raises at construction if it is absent and
        # OPENAI_API_KEY is unset. Ollama has no authentication and ignores the value.
        #
        # A hardcoded placeholder rather than read from the environment, deliberately in both
        # directions. Reading OPENAI_API_KEY would (a) fail confusingly for the many students
        # who do not have one, in a workshop that never contacts OpenAI, and (b) put a real
        # credential in an Authorization header sent to `base_url` — which is env-configurable,
        # so pointing MELLUM_BASE_URL elsewhere would leak the key there. "agentfix" is not a
        # secret; it is a protocol field the server discards.
        api_key="agentfix",  # type: ignore[arg-type]
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        extra_body={
            # `max_tokens` goes through extra_body rather than ChatOpenAI's own argument, and
            # this is not stylistic. ChatOpenAI's `max_tokens` field is aliased to OpenAI's
            # newer `max_completion_tokens`, and that is the key it puts on the wire —
            # which Ollama's /v1 endpoint ignores. Measured, asking for 8 tokens:
            #
            #     max_completion_tokens=8  -> 692 tokens generated   (ignored)
            #     max_tokens=8             ->   8 tokens generated   (honoured)
            #
            # Passing it the framework's way would silently remove the cap on a single reply,
            # which is the ceiling on how large a file write_file can emit in one turn.
            "max_tokens": config.max_tokens,
            # Ollama's /v1 endpoint DROPS this too (measured: `ollama ps` still says 4096). It
            # is kept because vLLM and Ollama's native /api/chat both honour it; on Ollama the
            # context comes from the derived model instead — see Modelfile. Switching this file
            # to `ChatOllama` would fix it, at the cost of no longer matching the original's
            # wire format. That trade is the subject of an exercise.
            "options": {"num_ctx": config.num_ctx},
        },
    )
