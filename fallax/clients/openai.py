"""OpenAI LLM client."""

from __future__ import annotations

import openai


class OpenAIClient:
    """LLM client using the OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key)
        self._max_tokens = max_tokens

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
