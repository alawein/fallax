"""Multi-provider LLM client factory."""

from __future__ import annotations

import os

from ..client import AnthropicClient, LLMClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_client(
    provider: str,
    *,
    api_key: str | None = None,
    max_tokens: int = 4096,
    base_url: str | None = None,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: One of 'anthropic', 'openai', 'openrouter', 'gemini', 'ollama'.
        api_key: API key (not needed for ollama).
        max_tokens: Maximum tokens in response.
        base_url: Custom API base URL (ollama and openrouter).
    """
    name = provider.lower()
    if name == "anthropic":
        return AnthropicClient(api_key=api_key, max_tokens=max_tokens)
    if name == "openai":
        from .openai import OpenAIClient

        return OpenAIClient(api_key=api_key, max_tokens=max_tokens, base_url=base_url)
    if name == "openrouter":
        # OpenRouter is OpenAI-API-compatible. Use the OpenAI client with the
        # OpenRouter base URL; model slugs take the form '<provider>/<model>'
        # (e.g. 'anthropic/claude-sonnet-4.6', 'openai/gpt-4o-mini').
        from .openai import OpenAIClient

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        return OpenAIClient(
            api_key=key,
            max_tokens=max_tokens,
            base_url=base_url or OPENROUTER_BASE_URL,
        )
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
