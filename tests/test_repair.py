import pytest

from reasonbench.models import RepairResult
from reasonbench.repair import SelfRepairTester
from tests.conftest import MockClient, make_result


@pytest.fixture()
def tester():
    client = MockClient(
        default="I was wrong because I assumed X. The correct answer is Y."
    )
    return SelfRepairTester(client=client)


@pytest.fixture()
def failed_results():
    return [
        make_result(
            prompt_id="f1",
            score=7,
            reasoning_flawed=True,
            prompt_text="hard prompt 1",
            answer="wrong answer 1",
            model_name="model-a",
        ),
        make_result(
            prompt_id="f2",
            score=6,
            reasoning_flawed=True,
            prompt_text="hard prompt 2",
            answer="wrong answer 2",
            model_name="model-b",
        ),
        make_result(
            prompt_id="ok",
            score=1,
            reasoning_flawed=False,
            prompt_text="easy prompt",
            answer="correct",
            model_name="model-a",
        ),
    ]


class TestSelfRepairTester:
    def test_test_repair_returns_repair_result(self, tester):
        result = tester.test_repair(
            prompt_text="test prompt",
            previous_answer="wrong",
            model="model-a",
        )
        assert isinstance(result, RepairResult)

    def test_repair_result_fields(self, tester):
        result = tester.test_repair(
            prompt_text="my prompt",
            previous_answer="my wrong answer",
            model="test-model",
        )
        assert result.model_name == "test-model"
        assert result.prompt_text == "my prompt"
        assert result.original_answer == "my wrong answer"
        assert len(result.repaired_answer) > 0
        assert len(result.repair_reasoning) > 0
        assert result.is_fixed is None  # not yet judged

    def test_repair_calls_client(self, tester):
        tester.test_repair("prompt", "answer", model="m")
        assert len(tester._client.calls) == 1
        prompt_sent = tester._client.calls[0][0]
        assert "previous answer was incorrect" in prompt_sent.lower()
        assert "prompt" in prompt_sent
        assert "answer" in prompt_sent

    def test_repair_uses_specified_model(self, tester):
        tester.test_repair("p", "a", model="my-model")
        assert tester._client.calls[0][1] == "my-model"

    def test_repair_batch_only_failed(self, tester, failed_results):
        repairs = tester.test_repair_batch(failed_results)
        # Only 2 results have reasoning_flawed=True
        assert len(repairs) == 2

    def test_repair_batch_returns_repair_results(self, tester, failed_results):
        repairs = tester.test_repair_batch(failed_results)
        assert all(isinstance(r, RepairResult) for r in repairs)

    def test_repair_batch_preserves_model_names(self, tester, failed_results):
        repairs = tester.test_repair_batch(failed_results)
        models = {r.model_name for r in repairs}
        assert models == {"model-a", "model-b"}

    def test_repair_batch_empty_when_no_failures(self, tester):
        good_results = [
            make_result(prompt_id="ok", score=0, reasoning_flawed=False),
        ]
        repairs = tester.test_repair_batch(good_results)
        assert repairs == []

    def test_repair_extracts_answer(self, tester):
        result = tester.test_repair("p", "a", model="m")
        # MockClient returns "I was wrong because I assumed X. The correct answer is Y."
        assert (
            result.repair_reasoning == result.repaired_answer
        )  # full response used as both
