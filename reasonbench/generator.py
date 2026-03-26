"""Prompt generator using parameter banks and templates."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .models import Prompt
from .templates import DISTRIBUTION, TemplateRegistry

TEMPLATE_DIFFICULTY: dict[str, int] = {
    "implicit_assumption_trap": 6,
    "contradictory_constraints": 7,
    "false_analogy_trap": 7,
    "recursive_definition_break": 8,
    "multi_step_dependency": 7,
    "edge_case_inversion": 5,
    "ambiguous_spec_trap": 6,
    "overconstrained_optimization": 8,
    "hidden_variable_trap": 6,
    "self_consistency_trap": 7,
}


class PromptGenerator:
    """Generates adversarial prompts from templates and parameter banks."""

    def __init__(
        self,
        registry: TemplateRegistry | None = None,
        params_dir: Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._registry = registry or TemplateRegistry()
        self._rng = random.Random(seed)
        self._params_dir = params_dir or Path(__file__).parent / "data"
        self._param_banks: dict[str, list[dict[str, str]]] = {}
        self._load_param_banks()

    @property
    def param_banks(self) -> dict[str, list[dict[str, str]]]:
        """Expose loaded parameter banks (read-only access)."""
        return self._param_banks

    def _load_param_banks(self) -> None:
        """Load parameter banks from JSON files in params_dir."""
        if not self._params_dir.is_dir():
            return
        for template_id in self._registry.template_ids():
            path = self._params_dir / f"{template_id}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._param_banks[template_id] = json.load(f)

    def generate_for_template(self, template_id: str, count: int) -> list[Prompt]:
        """Generate N prompts from a specific template."""
        bank = self._param_banks.get(template_id, [])
        if not bank:
            return []
        template = self._registry.get(template_id)
        difficulty = TEMPLATE_DIFFICULTY.get(template_id, 5)

        prompts: list[Prompt] = []
        for i in range(count):
            params = bank[i % len(bank)]
            prompt_text = self._registry.render(template_id, **params)
            prompts.append(
                Prompt(
                    failure_type=template.failure_target,
                    prompt_text=prompt_text,
                    difficulty=difficulty,
                    template_id=template_id,
                    parameters=params,
                )
            )
        return prompts

    def generate_batch(self, count: int = 100) -> list[Prompt]:
        """Generate a batch of prompts following distribution targets."""
        available = {
            tid: pct for tid, pct in DISTRIBUTION.items() if tid in self._param_banks
        }
        if not available:
            return []

        total_pct = sum(available.values())
        prompts: list[Prompt] = []
        for template_id, pct in available.items():
            n = max(1, round(pct * count / total_pct))
            prompts.extend(self.generate_for_template(template_id, n))

        self._rng.shuffle(prompts)
        return prompts[:count]
