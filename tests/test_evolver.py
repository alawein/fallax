import pytest

from reasonbench.evolver import PromptEvolver
from reasonbench.models import Assumption, Prompt
from reasonbench.taxonomy import FailureType
from tests.conftest import MockClient, make_result


@pytest.fixture()
def evolver():
    client = MockClient(default="A harder version of the prompt with more constraints.")
    return PromptEvolver(client=client, model="evolve-model")


@pytest.fixture()
def hard_results():
    return [
        make_result(
            prompt_id="hard-1",
            score=8,
            reasoning_flawed=True,
            prompt_text="original hard prompt",
            failure_type=FailureType.CONTRADICTION,
            assumptions=[Assumption(text="x is positive", justified=False)],
        ),
        make_result(
            prompt_id="hard-2",
            score=7,
            reasoning_flawed=True,
            prompt_text="another hard prompt",
            failure_type=FailureType.UNSTATED_ASSUMPTION,
        ),
        make_result(
            prompt_id="easy-1",
            score=2,
            reasoning_flawed=False,
            prompt_text="easy prompt",
            failure_type=FailureType.OVERGENERALIZATION,
        ),
    ]


class TestPromptEvolver:
    def test_evolve_returns_string(self, evolver):
        result = evolver.evolve("test prompt", "reasoning was flawed")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evolve_calls_client(self, evolver):
        evolver.evolve("my prompt", "my failure")
        assert len(evolver._client.calls) == 1
        prompt_sent = evolver._client.calls[0][0]
        assert "my prompt" in prompt_sent
        assert "my failure" in prompt_sent

    def test_evolve_uses_configured_model(self, evolver):
        evolver.evolve("p", "f")
        assert evolver._client.calls[0][1] == "evolve-model"

    def test_evolve_batch_filters_by_score(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert len(evolved) == 2  # only score 8 and 7

    def test_evolve_batch_returns_prompts(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert all(isinstance(p, Prompt) for p in evolved)

    def test_evolved_prompt_preserves_failure_type(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert evolved[0].failure_type == FailureType.CONTRADICTION
        assert evolved[1].failure_type == FailureType.UNSTATED_ASSUMPTION

    def test_evolved_prompt_has_no_template_id(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        assert all(p.template_id is None for p in evolved)

    def test_evolved_prompt_has_unique_ids(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=6)
        ids = [p.prompt_id for p in evolved]
        assert len(ids) == len(set(ids))

    def test_build_failure_analysis(self, hard_results):
        analysis = PromptEvolver.build_failure_analysis(hard_results[0])
        assert "8" in analysis  # score
        assert "critical" in analysis.lower()
        assert "x is positive" in analysis

    def test_build_failure_analysis_no_assumptions(self, hard_results):
        analysis = PromptEvolver.build_failure_analysis(hard_results[1])
        assert "7" in analysis
        assert "assumption" not in analysis.lower() or "Unjustified" not in analysis

    def test_evolve_batch_empty_when_no_hard_cases(self, evolver, hard_results):
        evolved = evolver.evolve_batch(hard_results, min_score=100)
        assert evolved == []
