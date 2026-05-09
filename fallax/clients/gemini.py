"""Google Gemini LLM client."""

from __future__ import annotations

import os

import google.generativeai as genai


class GeminiClient:
    """LLM client using the Google Generative AI API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if key:
            genai.configure(api_key=key)
        self._max_tokens = max_tokens

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self._max_tokens,
            ),
        )
        return str(response.text)
