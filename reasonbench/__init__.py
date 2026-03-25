"""ReasonBench — LLM Adversarial Reasoning Evaluation System."""

from .analyzer import Analyzer
from .client import AnthropicClient, LLMClient
from .clusterer import FailureClusterer
from .evaluator import Evaluator
from .evolver import PromptEvolver
from .generator import PromptGenerator
from .models import (
    Assumption,
    EvaluationResult,
    FailureRecord,
    ModelResponse,
    Prompt,
    RepairResult,
    RootCausePattern,
    ValidationResult,
)
from .pipeline import Pipeline
from .predictor import FailurePredictor
from .repair import SelfRepairTester
from .root_cause import RootCauseExtractor
from .runner import ModelRunner
from .scoring import Scorer
from .storage import JsonlStore
from .taxonomy import (
    FAILURE_CATEGORY_MAP,
    FailureCategory,
    FailureType,
    Severity,
    get_category,
)
from .templates import DISTRIBUTION, TemplateRegistry
from .validators import ValidatorPack

__all__ = [
    "Analyzer",
    "AnthropicClient",
    "Assumption",
    "DISTRIBUTION",
    "EvaluationResult",
    "Evaluator",
    "FAILURE_CATEGORY_MAP",
    "FailureCategory",
    "FailureClusterer",
    "FailurePredictor",
    "FailureRecord",
    "FailureType",
    "JsonlStore",
    "LLMClient",
    "ModelResponse",
    "ModelRunner",
    "Pipeline",
    "Prompt",
    "PromptEvolver",
    "PromptGenerator",
    "RepairResult",
    "RootCauseExtractor",
    "RootCausePattern",
    "Scorer",
    "SelfRepairTester",
    "Severity",
    "TemplateRegistry",
    "ValidationResult",
    "ValidatorPack",
    "get_category",
]
