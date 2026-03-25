"""Multi-round experiment orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from .analyzer import Analyzer
from .client import LLMClient
from .evaluator import Evaluator
from .evolver import PromptEvolver
from .generator import PromptGenerator
from .models import EvaluationResult, ExperimentRound, Prompt
from .repair import SelfRepairTester
from .root_cause import RootCauseExtractor
from .runner import ModelRunner
from .scoring import Scorer
from .storage import JsonlStore

logger = logging.getLogger(__name__)


class Experiment:
    """Orchestrates multi-round evaluation with evolution and reporting.

    Round 1 generates prompts via PromptGenerator. Each subsequent round
    evaluates evolved prompts from the prior round. After all rounds,
    root cause extraction runs on combined results and self-repair testing
    runs on the final round's failures.
    """

    def __init__(
        self,
        client: LLMClient,
        models: list[str],
        judge_model: str,
        output_dir: Path,
        evolve_model: str,
        params_dir: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._client = client
        self._models = models
        self._judge_model = judge_model
        self._output_dir = output_dir
        self._evolve_model = evolve_model
        self._params_dir = params_dir
        self._seed = seed

    def run(
        self,
        initial_count: int = 10,
        rounds: int = 3,
        min_score: int = 6,
    ) -> dict:
        """Run a multi-round experiment.

        Returns a dict with keys: rounds, root_cause_patterns,
        repair_results, total_prompts, total_failures.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        all_results: list[EvaluationResult] = []
        round_summaries: list[ExperimentRound] = []
        last_round_results: list[EvaluationResult] = []
        prompts: list[Prompt] | None = None

        for round_num in range(1, rounds + 1):
            round_path = self._output_dir / f"round_{round_num}.jsonl"

            if round_num == 1:
                generator = PromptGenerator(
                    params_dir=self._params_dir, seed=self._seed
                )
                prompts = generator.generate_batch(initial_count)

            if not prompts:
                break

            logger.info(
                "Round %d: evaluating %d prompts", round_num, len(prompts)
            )
            results = self._evaluate_prompts(prompts, round_path)
            all_results.extend(results)
            last_round_results = results

            analyzer = Analyzer(results)
            summary = analyzer.summary()
            hard_cases = analyzer.hard_cases(min_score)

            evolved: list[Prompt] = []
            if round_num < rounds and hard_cases:
                evolver = PromptEvolver(self._client, self._evolve_model)
                evolved = evolver.evolve_batch(results, min_score)

            round_summaries.append(
                ExperimentRound(
                    round_number=round_num,
                    prompts_evaluated=len(results),
                    avg_score=summary["avg_score"],
                    failure_rate=summary["failure_rate"],
                    hard_case_count=len(hard_cases),
                    evolved_count=len(evolved),
                )
            )

            prompts = evolved if evolved else None

        root_cause_patterns = []
        repair_results = []
        if all_results:
            extractor = RootCauseExtractor(all_results)
            root_cause_patterns = extractor.extract_patterns(
                min_frequency=1
            )
            tester = SelfRepairTester(self._client)
            repair_results = tester.test_repair_batch(last_round_results)

        return {
            "rounds": round_summaries,
            "root_cause_patterns": root_cause_patterns,
            "repair_results": repair_results,
            "total_prompts": sum(
                r.prompts_evaluated for r in round_summaries
            ),
            "total_failures": sum(
                1
                for r in all_results
                if r.validation.reasoning_flawed
            ),
        }

    def _evaluate_prompts(
        self, prompts: list[Prompt], output_path: Path
    ) -> list[EvaluationResult]:
        """Evaluate a list of prompts through the full pipeline."""
        runner = ModelRunner(self._client, self._models)
        evaluator = Evaluator(self._client, self._judge_model)
        store = JsonlStore(output_path)

        results: list[EvaluationResult] = []
        for prompt in prompts:
            model_responses = runner.run(prompt.prompt_text)
            first_response = next(iter(model_responses.values()))
            validation = evaluator.evaluate(
                prompt=prompt.prompt_text,
                answer=first_response.answer,
                reasoning=first_response.reasoning,
            )

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

            result = EvaluationResult(
                prompt_id=prompt.prompt_id,
                failure_type=prompt.failure_type,
                prompt_text=prompt.prompt_text,
                models=model_responses,
                validation=validation,
                score=score,
                severity=severity,
            )
            store.append(result)
            results.append(result)

        return results
