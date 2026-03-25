"""Model runner for multi-model evaluation."""

from __future__ import annotations

from .client import LLMClient
from .models import ModelResponse


class ModelRunner:
    """Runs prompts against multiple models and collects responses."""

    def __init__(self, client: LLMClient, models: list[str]) -> None:
        self._client = client
        self._models = models

    def run(self, prompt_text: str) -> dict[str, ModelResponse]:
        """Run a prompt against all configured models."""
        results: dict[str, ModelResponse] = {}
        for model_name in self._models:
            response_text = self._client.complete(
                prompt_text, model=model_name
            )
            results[model_name] = ModelResponse(
                model_name=model_name,
                answer=self._extract_answer(response_text),
                reasoning=response_text,
            )
        return results

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract the final answer from a model response.

        Looks for a line starting with 'ANSWER:'. Falls back to last
        non-empty line.
        """
        if not response.strip():
            return ""
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("ANSWER:"):
                return stripped[len("ANSWER:"):].strip()
        # Fallback: last non-empty line
        for line in reversed(response.split("\n")):
            if line.strip():
                return line.strip()
        return response.strip()
