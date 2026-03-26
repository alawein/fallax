import json
from pathlib import Path

import pytest

from reasonbench.generator import TEMPLATE_DIFFICULTY, PromptGenerator
from reasonbench.models import Prompt
from reasonbench.taxonomy import FailureType


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    """Create a minimal parameter bank for testing."""
    data = [
        {"rule_a": "x > 0", "rule_b": "double it", "edge_case_input": "x = -1"},
        {"rule_a": "len < 5", "rule_b": "reverse", "edge_case_input": "empty string"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))

    data2 = [
        {"constraint_1": "a > 10", "constraint_2": "a < 5", "constraint_3": "a > 0"},
    ]
    (tmp_path / "contradictory_constraints.json").write_text(json.dumps(data2))

    return tmp_path


class TestPromptGenerator:
    def test_load_param_banks(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir)
        banks = gen.param_banks
        assert "implicit_assumption_trap" in banks
        assert len(banks["implicit_assumption_trap"]) == 2

    def test_generate_single(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=1)
        assert len(prompts) == 1
        p = prompts[0]
        assert isinstance(p, Prompt)
        assert p.failure_type == FailureType.UNSTATED_ASSUMPTION
        assert p.template_id == "implicit_assumption_trap"
        assert "x > 0" in p.prompt_text or "len < 5" in p.prompt_text

    def test_generate_wraps_around_bank(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=5)
        assert len(prompts) == 5
        param_sets = [p.parameters for p in prompts]
        assert len(set(p["rule_a"] for p in param_sets)) <= 2

    def test_generate_batch_respects_count(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        batch = gen.generate_batch(count=3)
        assert len(batch) == 3

    def test_generate_batch_uses_available_templates_only(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        batch = gen.generate_batch(count=10)
        template_ids = {p.template_id for p in batch}
        assert template_ids <= {"implicit_assumption_trap", "contradictory_constraints"}

    def test_difficulty_from_template_type(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=1)
        assert prompts[0].difficulty == TEMPLATE_DIFFICULTY["implicit_assumption_trap"]

    def test_empty_bank_returns_empty(self, tmp_path):
        gen = PromptGenerator(params_dir=tmp_path)
        prompts = gen.generate_for_template("implicit_assumption_trap", count=5)
        assert prompts == []

    def test_seed_reproducibility(self, params_dir):
        gen1 = PromptGenerator(params_dir=params_dir, seed=42)
        gen2 = PromptGenerator(params_dir=params_dir, seed=42)
        batch1 = gen1.generate_batch(count=5)
        batch2 = gen2.generate_batch(count=5)
        assert [p.prompt_text for p in batch1] == [p.prompt_text for p in batch2]

    def test_unique_prompt_ids(self, params_dir):
        gen = PromptGenerator(params_dir=params_dir, seed=42)
        batch = gen.generate_batch(count=5)
        ids = [p.prompt_id for p in batch]
        assert len(ids) == len(set(ids))


class TestTemplateDifficulty:
    def test_covers_all_templates(self):
        from reasonbench.templates import TEMPLATES

        template_ids = {t.template_id for t in TEMPLATES}
        assert set(TEMPLATE_DIFFICULTY.keys()) == template_ids

    def test_values_in_range(self):
        for difficulty in TEMPLATE_DIFFICULTY.values():
            assert 1 <= difficulty <= 10
