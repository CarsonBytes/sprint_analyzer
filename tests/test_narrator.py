"""
Tests for the LLM backend functions.

These tests exercise:
  - env-based backend selection (LLM_PROVIDER)
  - the Anthropic call shape (mocked SDK)
  - the OpenAI-compatible call shape (mocked SDK)
  - missing-API-key error handling for both providers
  - model resolution (explicit arg > env > built-in default)
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from sprint_analyzer import narrator


# ---------- Provider selection ----------

class TestProviderSelection:
    def test_anthropic_is_default(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert narrator._active_provider() == "anthropic"

    @pytest.mark.parametrize("value", ["openai", "openai_compatible", "deepseek", "gpt_api_free"])
    def test_openai_aliases(self, monkeypatch, value):
        monkeypatch.setenv("LLM_PROVIDER", value)
        assert narrator._active_provider() == "openai"

    def test_unknown_provider_falls_back_to_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "magic-cloud")
        assert narrator._active_provider() == "anthropic"

    def test_default_llm_fn_picks_anthropic(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert narrator._default_llm_fn() is narrator._call_anthropic

    def test_default_llm_fn_picks_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert narrator._default_llm_fn() is narrator._call_openai_compatible


# ---------- Model resolution ----------

class TestModelResolution:
    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "from-env")
        assert narrator._resolved_model("explicit") == "explicit"

    def test_anthropic_env_wins_over_fallback(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_MODEL", "from-anthropic-env")
        assert narrator._resolved_model() == "from-anthropic-env"

    def test_openai_env_used_when_provider_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-from-env")
        assert narrator._resolved_model() == "deepseek-from-env"

    def test_anthropic_env_ignored_when_provider_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("ANTHROPIC_MODEL", "should-not-be-used")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        assert narrator._resolved_model() == narrator.OPENAI_MODEL_FALLBACK


# ---------- Anthropic backend (mocked) ----------

class TestCallAnthropic:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            narrator._call_anthropic("sys", "user", "claude-sonnet-4-5")

    def test_calls_sdk_with_expected_payload(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        fake_block = MagicMock(type="text", text="Hello from Claude")
        fake_response = MagicMock(content=[fake_block])
        fake_messages = MagicMock()
        fake_messages.create.return_value = fake_response
        fake_client = MagicMock(messages=fake_messages)

        with patch.object(narrator, "_call_anthropic", wraps=narrator._call_anthropic):
            with patch("anthropic.Anthropic", return_value=fake_client) as ctor:
                out = narrator._call_anthropic("SYS", "USER", "claude-sonnet-4-5")

        assert out == "Hello from Claude"
        ctor.assert_called_once()
        call_kwargs = fake_messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-5"
        assert call_kwargs["system"] == "SYS"
        assert call_kwargs["messages"] == [{"role": "user", "content": "USER"}]


# ---------- OpenAI-compatible backend (mocked) ----------

class TestCallOpenAICompatible:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            narrator._call_openai_compatible("sys", "user", "deepseek-v3.2-exp")

    def test_calls_sdk_with_expected_payload_and_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

        fake_message = MagicMock(content="Hello from DeepSeek")
        fake_choice = MagicMock(message=fake_message)
        fake_response = MagicMock(choices=[fake_choice])
        fake_completions = MagicMock()
        fake_completions.create.return_value = fake_response
        fake_chat = MagicMock(completions=fake_completions)
        fake_client = MagicMock(chat=fake_chat)

        with patch("openai.OpenAI", return_value=fake_client) as ctor:
            out = narrator._call_openai_compatible("SYS", "USER", "deepseek-v3.2-exp")

        assert out == "Hello from DeepSeek"
        ctor_kwargs = ctor.call_args.kwargs
        assert ctor_kwargs["api_key"] == "sk-test"
        assert ctor_kwargs["base_url"] == "https://example.test/v1"

        call_kwargs = fake_completions.create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-v3.2-exp"
        # System + user role split
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "SYS"},
            {"role": "user",   "content": "USER"},
        ]

    def test_base_url_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        fake_message = MagicMock(content="ok")
        fake_response = MagicMock(choices=[MagicMock(message=fake_message)])
        fake_completions = MagicMock()
        fake_completions.create.return_value = fake_response
        fake_client = MagicMock(chat=MagicMock(completions=fake_completions))

        with patch("openai.OpenAI", return_value=fake_client) as ctor:
            narrator._call_openai_compatible("s", "u", "m")

        assert ctor.call_args.kwargs["base_url"] == narrator.OPENAI_BASE_URL_FALLBACK


# ---------- Backend label (used by the UI) ----------

class TestBackendLabel:
    def test_anthropic_label(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
        label = narrator.active_backend_label()
        assert "Anthropic" in label
        assert "claude-test" in label

    def test_openai_label_includes_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://foo.test/v1")
        label = narrator.active_backend_label()
        assert "OpenAI-compatible" in label
        assert "deepseek-test" in label
        assert "https://foo.test/v1" in label
