"""Data models for prompts, responses, validation, and evaluation results."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from .taxonomy import FailureType, Severity


class Prompt(BaseModel):
    """A generated adversarial prompt."""

    prompt_id: str = Field(default_factory=lambda: str(uuid4()))
    failure_type: FailureType
    prompt_text: str
    difficulty: int = Field(ge=1, le=10)
    template_id: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """A model's response to a prompt."""

    model_name: str
    answer: str
    reasoning: str
    is_correct: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Assumption(BaseModel):
    """An implicit assumption extracted from reasoning."""

    text: str
    justified: bool


class ValidationResult(BaseModel):
    """Combined output from all 5 validators."""

    reasoning_flawed: bool
    first_error_step: int | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    counterfactual_fail: bool = False
    adversarial_issues: list[str] = Field(default_factory=list)
    final_answer_correct: bool | None = None


class FailureRecord(BaseModel):
    """A single identified failure in a model's reasoning."""

    failure_type: FailureType
    step_of_failure: int | None = None
    reason: str
    severity: Severity


class EvaluationResult(BaseModel):
    """Complete evaluation result for one prompt across models."""

    prompt_id: str
    failure_type: FailureType
    prompt_text: str
    models: dict[str, ModelResponse]
    validation: ValidationResult
    score: int
    severity: Severity
    failures: list[FailureRecord] = Field(default_factory=list)


class RepairResult(BaseModel):
    """Result of a self-repair attempt."""

    model_name: str
    prompt_text: str
    original_answer: str
    repaired_answer: str
    repair_reasoning: str
    is_fixed: bool | None = None


class RootCausePattern(BaseModel):
    """A recurring failure pattern extracted from results."""

    pattern: str
    frequency: int = Field(ge=1)
    models_affected: list[str]
    example_prompt: str
    failure_types: list[str]


class ExperimentRound(BaseModel):
    """Metadata for one round of an experiment."""

    round_number: int = Field(ge=1)
    prompts_evaluated: int = Field(ge=0)
    avg_score: float
    failure_rate: float
    hard_case_count: int = Field(ge=0)
    evolved_count: int = Field(ge=0)
    repair_success_rate: float | None = None
