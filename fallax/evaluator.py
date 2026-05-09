"""Validator orchestrator — runs 5 validators via judge LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from .client import LLMClient
from .models import Assumption, ValidationResult
from .validators import ValidatorPack

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs the validator prompt pack against model responses via a judge LLM.

    Each validator prompt is sent to the judge with a JSON schema instruction.
    Responses are parsed into structured data. Parse failures produce safe
    defaults (no crash, no false positives).
    """

    def __init__(self, client: LLMClient, judge_model: str) -> None:
        self._client = client
        self._judge_model = judge_model

    def evaluate(
        self,
        prompt: str,
        answer: str,
        reasoning: str,
        perturbation: str = "Change one key value in the input.",
    ) -> ValidationResult:
        """Run all 5 validators and assemble a ValidationResult."""
        critic = self._run_reasoning_critic(prompt, answer, reasoning)
        assumptions = self._run_assumption_extractor(reasoning)
        cf_fail = self._run_counterfactual_test(reasoning, perturbation)
        adv_issues = self._run_adversarial_challenger(reasoning)
        correct = self._run_truth_judge(prompt, answer)

        return ValidationResult(
            reasoning_flawed=critic["flawed"],
            first_error_step=critic.get("step"),
            assumptions=assumptions,
            counterfactual_fail=cf_fail,
            adversarial_issues=adv_issues,
            final_answer_correct=correct,
        )

    def _judge(self, validator_prompt: str, schema_hint: str) -> dict[str, Any]:
        """Call judge LLM and parse JSON response."""
        full_prompt = (
            f"{validator_prompt}\n\n"
            f"Respond ONLY with valid JSON matching: {schema_hint}"
        )
        response = self._client.complete(full_prompt, model=self._judge_model)
        return self._parse_json(response)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract JSON from text. Tries full parse, then embedded JSON."""
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try to find JSON object embedded in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}

    def _run_reasoning_critic(
        self, prompt: str, answer: str, reasoning: str
    ) -> dict[str, Any]:
        schema = (
            '{"reasoning_flawed": bool, "first_error_step": int|null, '
            '"explanation": "string"}'
        )
        validator_prompt = ValidatorPack.reasoning_critic(prompt, answer, reasoning)
        try:
            result = self._judge(validator_prompt, schema)
            return {
                "flawed": bool(result.get("reasoning_flawed", False)),
                "step": result.get("first_error_step"),
            }
        except Exception:
            logger.debug("Reasoning critic parse failed", exc_info=True)
            return {"flawed": False, "step": None}

    def _run_assumption_extractor(self, reasoning: str) -> list[Assumption]:
        schema = '{"assumptions": [{"text": "string", "justified": bool}]}'
        validator_prompt = ValidatorPack.assumption_extractor(reasoning)
        try:
            result = self._judge(validator_prompt, schema)
            return [
                Assumption(text=a["text"], justified=a["justified"])
                for a in result.get("assumptions", [])
            ]
        except Exception:
            logger.debug("Assumption extractor parse failed", exc_info=True)
            return []

    def _run_counterfactual_test(self, reasoning: str, perturbation: str) -> bool:
        schema = '{"holds": bool}'
        validator_prompt = ValidatorPack.counterfactual_test(reasoning, perturbation)
        try:
            result = self._judge(validator_prompt, schema)
            return not result.get("holds", True)
        except Exception:
            logger.debug("Counterfactual test parse failed", exc_info=True)
            return False

    def _run_adversarial_challenger(self, reasoning: str) -> list[str]:
        schema = '{"issues": ["string"], "robust": bool}'
        validator_prompt = ValidatorPack.adversarial_challenger(reasoning)
        try:
            result = self._judge(validator_prompt, schema)
            raw = result.get("issues", [])
            return [str(i) for i in raw] if isinstance(raw, list) else []
        except Exception:
            logger.debug("Adversarial challenger parse failed", exc_info=True)
            return []

    def _run_truth_judge(self, prompt: str, answer: str) -> bool | None:
        schema = '{"correct": bool}'
        validator_prompt = ValidatorPack.truth_judge(prompt, answer)
        try:
            result = self._judge(validator_prompt, schema)
            return result.get("correct")
        except Exception:
            logger.debug("Truth judge parse failed", exc_info=True)
            return None
