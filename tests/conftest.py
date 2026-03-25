"""Shared test fixtures for Phase 2 tests."""

from __future__ import annotations

import pytest


class MockClient:
    """Mock LLM client for testing. Matches LLMClient protocol."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "mock response",
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, model: str) -> str:
        self.calls.append((prompt, model))
        for keyword, response in self._responses.items():
            if keyword in prompt:
                return response
        return self._default


@pytest.fixture()
def mock_client():
    """A basic MockClient returning 'mock response'."""
    return MockClient()


JUDGE_RESPONSES: dict[str, str] = {
    "strict reasoning auditor": (
        '{"reasoning_flawed": true, "first_error_step": 2, '
        '"explanation": "Step 2 assumes without evidence"}'
    ),
    "implicit assumptions": (
        '{"assumptions": [{"text": "x is positive", "justified": false}]}'
    ),
    "Does the reasoning still hold": (
        '{"holds": false, "break_point": "Step 2 no longer valid"}'
    ),
    "prove the reasoning WRONG": (
        '{"issues": ["Step 2 has unjustified leap"], "robust": false}'
    ),
    "Is the final answer correct": '{"correct": false}',
}


@pytest.fixture()
def judge_client():
    """MockClient pre-loaded with judge LLM responses."""
    return MockClient(responses=JUDGE_RESPONSES)


MODEL_RESPONSE_TEXT = (
    "Let me work through this step by step.\n"
    "Step 1: I observe the input.\n"
    "Step 2: I apply the rule.\n"
    "ANSWER: The output is 42"
)


@pytest.fixture()
def model_client():
    """MockClient pre-loaded with model evaluation responses."""
    return MockClient(default=MODEL_RESPONSE_TEXT)
