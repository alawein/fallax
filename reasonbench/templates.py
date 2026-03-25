"""Adversarial prompt templates and registry."""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import FailureType


@dataclass(frozen=True)
class PromptTemplate:
    """A parameterizable adversarial prompt template."""

    template_id: str
    failure_target: FailureType
    template_text: str
    parameters: list[str]
    description: str


TEMPLATES: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        template_id="implicit_assumption_trap",
        failure_target=FailureType.UNSTATED_ASSUMPTION,
        template_text=(
            "A system processes inputs under the following rules:\n"
            "- Rule A: {rule_a}\n"
            "- Rule B: {rule_b}\n\n"
            "No other rules are specified.\n\n"
            "Question: What happens when {edge_case_input}?\n"
            "Explain your reasoning step by step."
        ),
        parameters=["rule_a", "rule_b", "edge_case_input"],
        description="Model invents missing rules",
    ),
    PromptTemplate(
        template_id="contradictory_constraints",
        failure_target=FailureType.IGNORED_CONSTRAINT,
        template_text=(
            "A process must satisfy ALL of the following:\n"
            "1. {constraint_1}\n"
            "2. {constraint_2}\n"
            "3. {constraint_3}\n\n"
            "Is it possible? If yes, provide an example. If no, prove why not."
        ),
        parameters=["constraint_1", "constraint_2", "constraint_3"],
        description="Model ignores one constraint",
    ),
    PromptTemplate(
        template_id="false_analogy_trap",
        failure_target=FailureType.PATTERN_MISAPPLICATION,
        template_text=(
            "Problem: This scenario looks similar to {well_known_problem}, "
            "but differs in the following way: {key_difference}\n\n"
            "Solve the problem."
        ),
        parameters=["well_known_problem", "key_difference"],
        description="Model applies wrong pattern",
    ),
    PromptTemplate(
        template_id="recursive_definition_break",
        failure_target=FailureType.CONTRADICTION,
        template_text=(
            "Define a function:\n"
            "- Base case: {base_case}\n"
            "- Recursive rule: {recursive_rule}\n\n"
            "Question: Is this function well-defined for all inputs? Justify."
        ),
        parameters=["base_case", "recursive_rule"],
        description="Misses contradiction between base and recursion",
    ),
    PromptTemplate(
        template_id="multi_step_dependency",
        failure_target=FailureType.MULTI_STEP_BREAK,
        template_text=(
            "Given:\n"
            "Step 1: {step1}\n"
            "Step 2 depends on Step 1: {step2}\n"
            "Step 3 depends on Step 2: {step3}\n\n"
            "Question: What is the final result?"
        ),
        parameters=["step1", "step2", "step3"],
        description="Error propagation unnoticed",
    ),
    PromptTemplate(
        template_id="edge_case_inversion",
        failure_target=FailureType.OVERGENERALIZATION,
        template_text=(
            "A rule works for all typical cases: {general_rule}\n\n"
            "Test it on this boundary case: {edge_case}\n\n"
            "Does the rule still hold?"
        ),
        parameters=["general_rule", "edge_case"],
        description="Model assumes general rule holds at boundaries",
    ),
    PromptTemplate(
        template_id="ambiguous_spec_trap",
        failure_target=FailureType.AMBIGUITY_FAILURE,
        template_text=(
            "A system is described as: {ambiguous_description}\n\n"
            "Question: What is the output for {input}?\n"
            "If assumptions are required, state them explicitly."
        ),
        parameters=["ambiguous_description", "input"],
        description="Model doesn't declare assumptions",
    ),
    PromptTemplate(
        template_id="overconstrained_optimization",
        failure_target=FailureType.PARTIAL_SATISFACTION,
        template_text=(
            "Maximize {objective} subject to: {constraints}\n\n"
            "Is there a solution? If so, find it. If not, explain why."
        ),
        parameters=["objective", "constraints"],
        description="Returns impossible solution",
    ),
    PromptTemplate(
        template_id="hidden_variable_trap",
        failure_target=FailureType.UNJUSTIFIED_ASSUMPTION,
        template_text=(
            "A result depends on variables X and Y.\n"
            "You are given: {partial_info}\n\n"
            "Find the result."
        ),
        parameters=["partial_info"],
        description="Model assumes missing variable",
    ),
    PromptTemplate(
        template_id="self_consistency_trap",
        failure_target=FailureType.INVALID_INFERENCE,
        template_text=(
            "Here is a reasoning chain:\n"
            "{reasoning_chain}\n\n"
            "Question: Is this reasoning correct? "
            "If not, identify the first incorrect step."
        ),
        parameters=["reasoning_chain"],
        description="Model agrees with flawed reasoning",
    ),
)

DISTRIBUTION: dict[str, int] = {
    "implicit_assumption_trap": 15,
    "contradictory_constraints": 15,
    "edge_case_inversion": 10,
    "false_analogy_trap": 10,
    "multi_step_dependency": 10,
    "recursive_definition_break": 10,
    "ambiguous_spec_trap": 10,
    "overconstrained_optimization": 10,
    "hidden_variable_trap": 5,
    "self_consistency_trap": 5,
}


class TemplateRegistry:
    """Registry for looking up and rendering prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {
            t.template_id: t for t in TEMPLATES
        }

    def get(self, template_id: str) -> PromptTemplate:
        """Get a template by ID. Raises KeyError if not found."""
        return self._templates[template_id]

    def list_all(self) -> list[PromptTemplate]:
        """Return all registered templates."""
        return list(self._templates.values())

    def list_by_failure_type(
        self, failure_type: FailureType
    ) -> list[PromptTemplate]:
        """Return templates targeting a specific failure type."""
        return [
            t
            for t in self._templates.values()
            if t.failure_target == failure_type
        ]

    def render(self, template_id: str, **params: str) -> str:
        """Render a template with given parameters.

        Raises KeyError if template not found or required parameter missing.
        """
        template = self.get(template_id)
        missing = set(template.parameters) - set(params.keys())
        if missing:
            raise KeyError(f"Missing parameters: {missing}")
        return template.template_text.format(**params)

    def template_ids(self) -> list[str]:
        """Return all template IDs."""
        return list(self._templates.keys())
