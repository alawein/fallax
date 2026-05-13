"""LLM client protocol and implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM API clients.

    Any object with a `complete(prompt, *, model) -> str` method satisfies this.
    After each `complete()` call, `served_model` reflects the model identifier
    the provider returned in the response (for direct providers, the same
    string as `model`; for gateways like OpenRouter, the gateway's resolution
    of the requested slug). It is None before the first call.
    """

    served_model: str | None

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        ...


class AnthropicClient:
    """LLM client using the Anthropic API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._max_tokens = max_tokens
        self.served_model: str | None = None

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        import anthropic

        message = self._client.messages.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self.served_model = message.model
        block = message.content[0]
        if isinstance(block, anthropic.types.TextBlock):
            return block.text
        raise ValueError(
            f"Expected TextBlock from model {model!r}, got {type(block).__name__!r}"
        )
