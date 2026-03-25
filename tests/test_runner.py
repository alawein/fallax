import pytest

from reasonbench.models import ModelResponse
from reasonbench.runner import ModelRunner
from tests.conftest import MockClient


class TestModelRunner:
    def test_run_single_model(self):
        client = MockClient(default="Step 1: think.\nANSWER: 42")
        runner = ModelRunner(client, models=["model-a"])
        results = runner.run("What is the answer?")
        assert len(results) == 1
        assert "model-a" in results
        resp = results["model-a"]
        assert isinstance(resp, ModelResponse)
        assert resp.model_name == "model-a"

    def test_run_dual_model(self):
        client = MockClient(default="I think the answer is yes.")
        runner = ModelRunner(client, models=["model-a", "model-b"])
        results = runner.run("prompt")
        assert len(results) == 2
        assert "model-a" in results
        assert "model-b" in results

    def test_reasoning_is_full_response(self):
        response = "Step 1: observe.\nStep 2: deduce.\nANSWER: 42"
        client = MockClient(default=response)
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.reasoning == response

    def test_answer_extracted_from_answer_line(self):
        response = "Step 1: think.\nANSWER: The result is 42"
        client = MockClient(default=response)
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.answer == "The result is 42"

    def test_answer_fallback_last_line(self):
        response = "Step 1: think.\nThe result is probably 42"
        client = MockClient(default=response)
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.answer == "The result is probably 42"

    def test_calls_client_with_correct_model(self):
        client = MockClient(default="response")
        runner = ModelRunner(client, models=["fast-model", "large-model"])
        runner.run("test prompt")
        models_called = [call[1] for call in client.calls]
        assert models_called == ["fast-model", "large-model"]

    def test_is_correct_defaults_to_none(self):
        client = MockClient(default="some answer")
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.is_correct is None

    def test_empty_response(self):
        client = MockClient(default="")
        runner = ModelRunner(client, models=["m"])
        resp = runner.run("prompt")["m"]
        assert resp.answer == ""
        assert resp.reasoning == ""
