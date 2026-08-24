"""Settings: where the model is, which model, and how to sample from it."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# Task fixtures and result files live in the repo, not in whatever directory the student
# happened to be standing in when they ran the CLI. `.parents[2]` climbs
# src/agentfix/ -> src/ -> repo root, derived rather than hardcoded so the path is right
# however the CLI was invoked.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Ollama's own API root — not the `/v1` compatibility endpoint. `ChatOllama` speaks the
# native protocol, which is the only one that honours `num_ctx` and `num_predict`; see
# llm/client.py for what the `/v1` endpoint silently discarded.
DEFAULT_BASE_URL = "http://localhost:11434"

# The model the agent talks to is the one derived by `ollama create -f Modelfile`, not the
# raw GGUF pull: only the derived model carries `num_ctx 16384`, because Ollama's /v1
# endpoint drops per-request `options`. See Modelfile for the measurement.
BASE_MODEL = "hf.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M"
DEFAULT_MODEL = "agentfix-mellum2"

# `agentfix doctor` fails if the loaded model reports less than this. A too-small context
# does not error — it silently truncates the middle of the agent's history, which looks like
# a stupid model rather than a misconfigured one.
MIN_CONTEXT_LENGTH = 16384


@dataclass(frozen=True)
class LLMConfig:
    """How to reach the model, and how to sample from it. Frozen: read-only after creation."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    # Conventional defaults for this class of model, not the result of a sweep on this
    # project. The trade-off if you change them: 0.0 makes eval numbers reproducible, while a
    # non-zero value gives a stuck model a way out of repeating itself.
    temperature: float = 0.6
    top_p: float = 0.95

    # Cap on ONE reply. Relevant because write_file must emit a complete file: this is the
    # ceiling on how large a file the agent can rewrite in a single turn. Reaches the server as
    # Ollama's `num_predict`.
    max_tokens: int = 1024

    # The context window the model is loaded with. Honoured now that the client speaks Ollama's
    # native API, which is why `agentfix doctor` can check it and expect to be obeyed.
    num_ctx: int = 16384

    @property
    def api_url(self) -> str:
        """`base_url` with a trailing `/v1` removed, if someone's environment still has one.

        Tolerance rather than a second endpoint: every request this project makes now goes to
        Ollama's native API, so a `MELLUM_BASE_URL` left over from the `/v1` days would
        otherwise produce URLs like `.../v1/api/ps`.
        """
        trimmed = self.base_url.rstrip("/")
        return trimmed[: -len("/v1")].rstrip("/") if trimmed.endswith("/v1") else trimmed

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Defaults, overridden by MELLUM_BASE_URL and MELLUM_MODEL if they are set."""
        return replace(
            cls(),
            base_url=os.environ.get("MELLUM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("MELLUM_MODEL", DEFAULT_MODEL),
        )
