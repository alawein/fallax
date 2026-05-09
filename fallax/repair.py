"""Self-repair testing — checks if models can fix their own reasoning failures."""

from __future__ import annotations

from .client import LLMClient
from .models import EvaluationResult, RepairResult


class SelfRepairTester:
    """Tests model ability to correct its own reasoning failures.

    Sends the original prompt and wrong answer back to the model,
    asking it to fix its reasoning. Tracks whether the model can
    self-correct.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def test_repair(
        self, prompt_text: str, previous_answer: str, model: str
    ) -> RepairResult:
        """Give a model another chance to fix its answer."""
        repair_prompt = (
            "Your previous answer was incorrect.\n\n"
            f"PROMPT: {prompt_text}\n"
            f"PREVIOUS ANSWER: {previous_answer}\n\n"
            "Fix your answer and explain what was wrong."
        )
        response = self._client.complete(repair_prompt, model=model)
        return RepairResult(
            model_name=model,
            prompt_text=prompt_text,
            original_answer=previous_answer,
            repaired_answer=response,
            repair_reasoning=response,
        )

    def test_repair_batch(self, results: list[EvaluationResult]) -> list[RepairResult]:
        """Test repair on all results with flawed reasoning."""
        repairs: list[RepairResult] = []
        for r in results:
            if not r.validation.reasoning_flawed:
                continue
            for model_name, response in r.models.items():
                repair = self.test_repair(r.prompt_text, response.answer, model_name)
                repairs.append(repair)
        return repairs
