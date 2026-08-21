"""The real client's configuration. Constructed, never called — no network here."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agentfix.config import LLMConfig
from agentfix.llm.client import make_chat_model


class TestMakeChatModel(unittest.TestCase):
    def test_the_config_reaches_the_client(self):
        model = make_chat_model(LLMConfig(base_url="http://x/v1", model="m", temperature=0.1))
        self.assertEqual(model.openai_api_base, "http://x/v1")
        self.assertEqual(model.model_name, "m")
        self.assertEqual(model.temperature, 0.1)

    def test_max_tokens_is_sent_as_max_tokens_and_not_the_frameworks_alias(self):
        """A regression guard, not a style preference.

        ChatOpenAI's own `max_tokens` argument is aliased to OpenAI's newer
        `max_completion_tokens`, and that is the key it puts on the wire. Ollama's /v1 ignores
        it. Measured, asking for 8 tokens: via the alias, 692 were generated; via `max_tokens`,
        8 were. "Simplifying" this to the framework's argument silently removes the ceiling on
        how large a file write_file can emit in one turn.
        """
        model = make_chat_model(LLMConfig(max_tokens=1024))
        self.assertEqual((model.extra_body or {})["max_tokens"], 1024)
        self.assertIsNone(model.max_tokens, "the aliased field must stay unset")

    def test_num_ctx_is_sent_for_the_servers_that_honour_it(self):
        model = make_chat_model(LLMConfig(num_ctx=16384))
        self.assertEqual((model.extra_body or {})["options"]["num_ctx"], 16384)

    def test_the_api_key_is_a_placeholder_and_never_read_from_the_environment(self):
        """Pointing MELLUM_BASE_URL elsewhere must not leak a real OpenAI credential."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-real-secret"}):
            model = make_chat_model(LLMConfig())
        self.assertEqual(model.openai_api_key.get_secret_value(), "agentfix")

    def test_env_configuration_is_picked_up_when_no_config_is_passed(self):
        with mock.patch.dict(os.environ, {"MELLUM_MODEL": "other-model"}):
            self.assertEqual(make_chat_model().model_name, "other-model")
