from unittest.mock import MagicMock

import anthropic
import pytest

from reasonbench.client import AnthropicClient, LLMClient
from tests.conftest import MockClient


class TestLLMClientProtocol:
    def test_mock_client_satisfies_protocol(self):
        client: LLMClient = MockClient()
        result = client.complete("hello", model="test")
        assert isinstance(result, str)


class TestMockClient:
    def test_default_response(self):
        client = MockClient(default="hello world")
        assert client.complete("anything", model="m") == "hello world"

    def test_keyword_response(self):
        client = MockClient(responses={"foo": "bar"})
        assert client.complete("contains foo here", model="m") == "bar"

    def test_records_calls(self):
        client = MockClient()
        client.complete("prompt1", model="model-a")
        client.complete("prompt2", model="model-b")
        assert len(client.calls) == 2
        assert client.calls[0] == ("prompt1", "model-a")
        assert client.calls[1] == ("prompt2", "model-b")

    def test_keyword_priority_first_match(self):
        client = MockClient(responses={"alpha": "first", "beta": "second"})
        result = client.complete("has alpha and beta", model="m")
        assert result == "first"


class TestAnthropicClient:
    def _make_client(self, mock_anthropic: MagicMock) -> AnthropicClient:
        client = AnthropicClient.__new__(AnthropicClient)
        client._client = mock_anthropic
        client._max_tokens = 4096
        return client

    def test_complete_calls_api(self):
        mock_anthropic = MagicMock()
        block = anthropic.types.TextBlock.model_construct(
            type="text", text="test response"
        )
        mock_message = MagicMock()
        mock_message.content = [block]
        mock_anthropic.messages.create.return_value = mock_message

        result = self._make_client(mock_anthropic).complete(
            "test prompt", model="test-model"
        )

        assert result == "test response"
        mock_anthropic.messages.create.assert_called_once_with(
            model="test-model",
            max_tokens=4096,
            messages=[{"role": "user", "content": "test prompt"}],
        )

    def test_complete_extracts_first_content_block(self):
        mock_anthropic = MagicMock()
        block1 = anthropic.types.TextBlock.model_construct(
            type="text", text="first block"
        )
        block2 = anthropic.types.TextBlock.model_construct(
            type="text", text="second block"
        )
        mock_message = MagicMock()
        mock_message.content = [block1, block2]
        mock_anthropic.messages.create.return_value = mock_message

        result = self._make_client(mock_anthropic).complete("test", model="m")
        assert result == "first block"

    def test_complete_raises_for_non_text_block(self):
        mock_anthropic = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock()]  # plain MagicMock is not a TextBlock
        mock_anthropic.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="Expected TextBlock"):
            self._make_client(mock_anthropic).complete("test", model="test-model")
