"""Multi-provider LLM client factory."""

from __future__ import annotations

from ..client import AnthropicClient, LLMClient


def create_client(
    provider: str,
    *,
    api_key: str | None = None,
    max_tokens: int = 4096,
    base_url: str | None = None,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: One of 'anthropic', 'openai', 'gemini', 'ollama'.
        api_key: API key (not needed for ollama).
        max_tokens: Maximum tokens in response.
        base_url: Custom API base URL (ollama only).
    """
    name = provider.lower()
    if name == "anthropic":
        return AnthropicClient(api_key=api_key, max_tokens=max_tokens)
    if name == "openai":
        from .openai import OpenAIClient

        return OpenAIClient(api_key=api_key, max_tokens=max_tokens)
    if name == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(api_key=api_key, max_tokens=max_tokens)
    if name == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(
            base_url=base_url or "http://localhost:11434",
        )
    raise ValueError(f"Unknown provider: {provider!r}")


__all__ = [
    "create_client",
]
