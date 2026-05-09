"""Prompt evolution — generates harder variants of high-scoring failures."""

from __future__ import annotations

from .client import LLMClient
from .models import EvaluationResult, Prompt


class PromptEvolver:
    """Takes hard-case prompts and evolves them into harder versions via LLM."""

    def __init__(self, client: LLMClient, model: str) -> None:
        self._client = client
        self._model = model

    def evolve(self, prompt_text: str, failure_analysis: str) -> str:
        """Generate a harder version of a failed prompt."""
        evolution_prompt = (
            "Given this prompt and its failure:\n\n"
            f"PROMPT: {prompt_text}\n"
            f"FAILURE: {failure_analysis}\n\n"
            "Generate a HARDER version that:\n"
            "- increases ambiguity OR\n"
            "- adds constraint interaction OR\n"
            "- introduces edge cases\n\n"
            "Return only the new prompt."
        )
        return self._client.complete(evolution_prompt, model=self._model)

    def evolve_batch(
        self, results: list[EvaluationResult], min_score: int = 6
    ) -> list[Prompt]:
        """Evolve all hard cases from a result set."""
        hard_cases = [r for r in results if r.score >= min_score]
        evolved: list[Prompt] = []
        for r in hard_cases:
            analysis = self.build_failure_analysis(r)
            new_text = self.evolve(r.prompt_text, analysis)
            evolved.append(
                Prompt(
                    failure_type=r.failure_type,
                    prompt_text=new_text,
                    difficulty=min(10, r.score),
                    template_id=None,
                )
            )
        return evolved

    @staticmethod
    def build_failure_analysis(result: EvaluationResult) -> str:
        """Build a failure analysis string from an evaluation result."""
        parts = [f"Score: {result.score} ({result.severity.value})"]
        if result.validation.reasoning_flawed:
            parts.append(
                f"Reasoning flawed at step {result.validation.first_error_step}"
            )
        unjustified = [a for a in result.validation.assumptions if not a.justified]
        if unjustified:
            parts.append(
                "Unjustified assumptions: " + ", ".join(a.text for a in unjustified)
            )
        if result.validation.counterfactual_fail:
            parts.append("Failed counterfactual test")
        if result.validation.adversarial_issues:
            parts.append(
                f"Adversarial issues: {len(result.validation.adversarial_issues)}"
            )
        return ". ".join(parts)
