"""OpenAI LLM client.

Also serves as the client for OpenAI-compatible gateways like OpenRouter.
Pass a `base_url` to redirect to the gateway; the v1 chat-completions shape
is shared. After each call, `served_model` reflects the model the provider
returned (which may differ from the requested slug when going through a
gateway).
"""

from __future__ import annotations

import openai


class OpenAIClient:
    """LLM client using the OpenAI v1 chat-completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._max_tokens = max_tokens
        self.served_model: str | None = None

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self.served_model = response.model
        content = response.choices[0].message.content
        return content if content is not None else ""
