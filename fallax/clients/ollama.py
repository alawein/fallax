"""Ollama local LLM client."""

from __future__ import annotations

import requests


class OllamaClient:
    """LLM client for local models via Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._base_url = base_url
        self.served_model: str | None = None

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        # Ollama echoes the model in its response body; fall back to the
        # requested name if absent for older daemons.
        self.served_model = body.get("model", model)
        return str(body["response"])
