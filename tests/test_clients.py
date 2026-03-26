"""Tests for multi-provider LLM clients and factory."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from reasonbench.client import LLMClient
from reasonbench.clients import create_client
from reasonbench.clients.ollama import OllamaClient


@pytest.fixture()
def mock_api_key():
    """Provide a placeholder API key for mocked client tests."""
    return "placeholder"


@pytest.fixture()
def mock_openai_module():
    """Inject a fake openai module into sys.modules."""
    fake = MagicMock()
    with patch.dict(sys.modules, {"openai": fake}):
        mod = importlib.import_module("reasonbench.clients.openai")
        importlib.reload(mod)
        yield fake, mod


@pytest.fixture()
def mock_genai_module():
    """Inject a fake google.generativeai module into sys.modules."""
    fake = MagicMock()
    with patch.dict(sys.modules, {"google": MagicMock(), "google.generativeai": fake}):
        mod = importlib.import_module("reasonbench.clients.gemini")
        importlib.reload(mod)
        yield fake, mod


class TestOpenAIClient:
    def test_satisfies_protocol(self, mock_openai_module, mock_api_key):
        _fake, mod = mock_openai_module
        client = mod.OpenAIClient(api_key=mock_api_key)
        assert isinstance(client, LLMClient)

    def test_complete_calls_api(self, mock_openai_module, mock_api_key):
        fake, mod = mock_openai_module
        client = mod.OpenAIClient(api_key=mock_api_key)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        fake.OpenAI.return_value.chat.completions.create.return_value = mock_response
        result = client.complete("prompt", model="gpt-4o")
        assert result == "hello"

    def test_max_tokens_configurable(self, mock_openai_module, mock_api_key):
        _fake, mod = mock_openai_module
        client = mod.OpenAIClient(api_key=mock_api_key, max_tokens=1024)
        assert client._max_tokens == 1024


class TestGeminiClient:
    def test_satisfies_protocol(self, mock_genai_module, mock_api_key):
        _fake, mod = mock_genai_module
        client = mod.GeminiClient(api_key=mock_api_key)
        assert isinstance(client, LLMClient)

    def test_complete_calls_api(self, mock_genai_module, mock_api_key):
        _fake, mod = mock_genai_module
        client = mod.GeminiClient(api_key=mock_api_key)
        mock_response = MagicMock()
        mock_response.text = "gemini says hi"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mod.genai.GenerativeModel.return_value = mock_model
        result = client.complete("prompt", model="gemini-2.5-pro")
        assert result == "gemini says hi"

    def test_max_tokens_configurable(self, mock_genai_module, mock_api_key):
        _fake, mod = mock_genai_module
        client = mod.GeminiClient(api_key=mock_api_key, max_tokens=2048)
        assert client._max_tokens == 2048


class TestOllamaClient:
    def test_satisfies_protocol(self):
        client = OllamaClient()
        assert isinstance(client, LLMClient)

    def test_complete_calls_api(self):
        with patch("reasonbench.clients.ollama.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "local model says hi"}
            mock_response.raise_for_status = MagicMock()
            mock_requests.post.return_value = mock_response
            client = OllamaClient()
            result = client.complete("prompt", model="llama3")
        assert result == "local model says hi"

    def test_custom_base_url(self):
        client = OllamaClient(base_url="http://custom:11434")
        assert client._base_url == "http://custom:11434"

    def test_default_base_url(self):
        client = OllamaClient()
        assert client._base_url == "http://localhost:11434"


class TestCreateClient:
    def test_anthropic_provider(self, mock_api_key):
        with patch("reasonbench.clients.AnthropicClient"):
            client = create_client("anthropic", api_key=mock_api_key)
        assert client is not None

    def test_openai_provider(self, mock_openai_module, mock_api_key):
        client = create_client("openai", api_key=mock_api_key)
        assert client is not None

    def test_gemini_provider(self, mock_genai_module, mock_api_key):
        client = create_client("gemini", api_key=mock_api_key)
        assert client is not None

    def test_ollama_provider(self):
        client = create_client("ollama")
        assert isinstance(client, OllamaClient)

    def test_ollama_with_base_url(self):
        client = create_client("ollama", base_url="http://gpu:11434")
        assert isinstance(client, OllamaClient)
        assert client._base_url == "http://gpu:11434"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client("unknown")

    def test_case_insensitive(self, mock_api_key):
        with patch("reasonbench.clients.AnthropicClient"):
            client = create_client("Anthropic", api_key=mock_api_key)
        assert client is not None
