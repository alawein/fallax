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

    def complete(self, prompt: str, *, model: str) -> str:
        """Send a prompt and return the text response."""
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return str(response.json()["response"])
