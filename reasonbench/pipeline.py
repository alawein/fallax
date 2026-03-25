"""Main evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from .client import LLMClient
from .evaluator import Evaluator
from .generator import PromptGenerator
from .models import EvaluationResult
from .runner import ModelRunner
from .scoring import Scorer
from .storage import JsonlStore

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates prompt generation, model evaluation, validation, and scoring."""

    def __init__(
        self,
        client: LLMClient,
        models: list[str],
        judge_model: str,
        output_path: Path,
        params_dir: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._generator = PromptGenerator(
            params_dir=params_dir, seed=seed
        )
        self._runner = ModelRunner(client, models)
        self._evaluator = Evaluator(client, judge_model)
        self._store = JsonlStore(output_path)

    def run(self, count: int = 100) -> list[EvaluationResult]:
        """Run the full pipeline: generate -> evaluate -> validate -> score -> store."""
        prompts = self._generator.generate_batch(count)
        if not prompts:
            return []

        results: list[EvaluationResult] = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(
                "Evaluating prompt %d/%d [%s]",
                i,
                len(prompts),
                prompt.template_id,
            )

            # Run all models
            model_responses = self._runner.run(prompt.prompt_text)

            # Validate the first model's response
            first_response = next(iter(model_responses.values()))
            validation = self._evaluator.evaluate(
                prompt=prompt.prompt_text,
                answer=first_response.answer,
                reasoning=first_response.reasoning,
            )

            # Compute score
            unjustified = sum(
                1 for a in validation.assumptions if not a.justified
            )
            answers = {r.answer for r in model_responses.values()}
            disagreement = len(answers) > 1

            score = Scorer.compute_score(
                is_correct=validation.final_answer_correct or False,
                reasoning_flawed=validation.reasoning_flawed,
                assumption_errors=unjustified,
                counterfactual_fail=validation.counterfactual_fail,
                model_disagreement=disagreement,
            )
            severity = Scorer.severity(score)

            # Assemble and store result
            result = EvaluationResult(
                prompt_id=prompt.prompt_id,
                failure_type=prompt.failure_type,
                prompt_text=prompt.prompt_text,
                models=model_responses,
                validation=validation,
                score=score,
                severity=severity,
            )
            self._store.append(result)
            results.append(result)

            if score >= 6:
                logger.info(
                    "  -> CRITICAL failure (score=%d): %s",
                    score,
                    prompt.template_id,
                )

        return results
