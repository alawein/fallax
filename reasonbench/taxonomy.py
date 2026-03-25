"""Failure taxonomy for adversarial reasoning evaluation.

Tier 1: Six broad failure categories.
Tier 2: Ten granular failure types mapped to Tier 1 categories.
"""

from enum import Enum


class FailureCategory(str, Enum):
    """Tier 1 failure categories."""

    LOGIC_ERROR = "logic_error"
    ASSUMPTION_ERROR = "assumption_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    GENERALIZATION_ERROR = "generalization_error"
    AMBIGUITY_FAILURE = "ambiguity_failure"
    MULTI_STEP_BREAK = "multi_step_break"


class FailureType(str, Enum):
    """Tier 2 granular failure types."""

    CONTRADICTION = "contradiction"
    INVALID_INFERENCE = "invalid_inference"
    UNSTATED_ASSUMPTION = "unstated_assumption"
    UNJUSTIFIED_ASSUMPTION = "unjustified_assumption"
    IGNORED_CONSTRAINT = "ignored_constraint"
    PARTIAL_SATISFACTION = "partial_satisfaction"
    OVERGENERALIZATION = "overgeneralization"
    PATTERN_MISAPPLICATION = "pattern_misapplication"
    AMBIGUITY_FAILURE = "ambiguity_failure"
    MULTI_STEP_BREAK = "multi_step_break"


class Severity(str, Enum):
    """Score-based severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


FAILURE_CATEGORY_MAP: dict[FailureType, FailureCategory] = {
    FailureType.CONTRADICTION: FailureCategory.LOGIC_ERROR,
    FailureType.INVALID_INFERENCE: FailureCategory.LOGIC_ERROR,
    FailureType.UNSTATED_ASSUMPTION: FailureCategory.ASSUMPTION_ERROR,
    FailureType.UNJUSTIFIED_ASSUMPTION: FailureCategory.ASSUMPTION_ERROR,
    FailureType.IGNORED_CONSTRAINT: FailureCategory.CONSTRAINT_VIOLATION,
    FailureType.PARTIAL_SATISFACTION: FailureCategory.CONSTRAINT_VIOLATION,
    FailureType.OVERGENERALIZATION: FailureCategory.GENERALIZATION_ERROR,
    FailureType.PATTERN_MISAPPLICATION: FailureCategory.GENERALIZATION_ERROR,
    FailureType.AMBIGUITY_FAILURE: FailureCategory.AMBIGUITY_FAILURE,
    FailureType.MULTI_STEP_BREAK: FailureCategory.MULTI_STEP_BREAK,
}


def get_category(failure_type: FailureType) -> FailureCategory:
    """Look up the Tier 1 category for a Tier 2 failure type."""
    return FAILURE_CATEGORY_MAP[failure_type]
