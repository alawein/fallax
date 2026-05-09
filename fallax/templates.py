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
    PromptTemplate(
        template_id="temporal_ordering_trap",
        failure_target=FailureType.MULTI_STEP_BREAK,
        template_text=(
            "Events occur in the following order:\n"
            "{event_sequence}\n\n"
            "Question: {temporal_question}\n"
            "Consider the time dependencies carefully."
        ),
        parameters=["event_sequence", "temporal_question"],
        description="Misses implicit temporal dependencies between events",
    ),
    PromptTemplate(
        template_id="negation_scope_trap",
        failure_target=FailureType.INVALID_INFERENCE,
        template_text=(
            "Statement: {negated_statement}\n\n"
            "Question: Which of the following must be true?\n"
            "A) {option_a}\n"
            "B) {option_b}\n"
            "Explain your reasoning."
        ),
        parameters=["negated_statement", "option_a", "option_b"],
        description="Misinterprets the scope of negation",
    ),
    PromptTemplate(
        template_id="base_rate_neglect",
        failure_target=FailureType.UNJUSTIFIED_ASSUMPTION,
        template_text=(
            "Background: {base_rate_info}\n"
            "Observation: {observation}\n\n"
            "Question: What is the probability that {target_event}?\n"
            "Show your reasoning."
        ),
        parameters=["base_rate_info", "observation", "target_event"],
        description="Ignores base rates in probabilistic reasoning",
    ),
    PromptTemplate(
        template_id="survivorship_bias_trap",
        failure_target=FailureType.OVERGENERALIZATION,
        template_text=(
            "Data: {survivor_data}\n\n"
            "Question: Based on this data, {conclusion_question}\n"
            "Is this conclusion justified? Why or why not?"
        ),
        parameters=["survivor_data", "conclusion_question"],
        description="Draws conclusions from non-representative samples",
    ),
    PromptTemplate(
        template_id="modus_tollens_break",
        failure_target=FailureType.CONTRADICTION,
        template_text=(
            "Given:\n"
            "- If {antecedent}, then {consequent}\n"
            "- It is known that {negated_consequent}\n\n"
            "Question: What can be concluded about {antecedent}?"
        ),
        parameters=[
            "antecedent",
            "consequent",
            "negated_consequent",
        ],
        description="Fails to apply contrapositive reasoning correctly",
    ),
    PromptTemplate(
        template_id="scope_creep_trap",
        failure_target=FailureType.PARTIAL_SATISFACTION,
        template_text=(
            "Original task: {original_task}\n"
            "Additional request: {scope_addition}\n\n"
            "Complete both the original task and the addition. "
            "Show all work."
        ),
        parameters=["original_task", "scope_addition"],
        description="Loses track of original requirements when scope expands",
    ),
    PromptTemplate(
        template_id="anchoring_trap",
        failure_target=FailureType.UNJUSTIFIED_ASSUMPTION,
        template_text=(
            "Context: {anchor_info}\n"
            "Question: {estimation_question}\n\n"
            "Provide your best estimate with reasoning."
        ),
        parameters=["anchor_info", "estimation_question"],
        description="Irrelevant numbers bias quantitative reasoning",
    ),
    PromptTemplate(
        template_id="false_dichotomy_trap",
        failure_target=FailureType.IGNORED_CONSTRAINT,
        template_text=(
            "Situation: {situation}\n\n"
            "You must choose: {option_a} or {option_b}.\n"
            "Which is better and why?"
        ),
        parameters=["situation", "option_a", "option_b"],
        description="Accepts false binary when other options exist",
    ),
    PromptTemplate(
        template_id="composition_fallacy",
        failure_target=FailureType.OVERGENERALIZATION,
        template_text=(
            "Each part has property: {part_property}\n"
            "Question: Does the whole system of {num_parts} parts "
            "also have this property? Explain."
        ),
        parameters=["part_property", "num_parts"],
        description="Assumes properties of parts apply to the whole",
    ),
    PromptTemplate(
        template_id="conjunction_fallacy",
        failure_target=FailureType.INVALID_INFERENCE,
        template_text=(
            "Profile: {person_profile}\n\n"
            "Which is more probable?\n"
            "A) {broad_category}\n"
            "B) {narrow_conjunction}\n"
            "Explain your answer."
        ),
        parameters=["person_profile", "broad_category", "narrow_conjunction"],
        description="Rates P(A and B) higher than P(A)",
    ),
    PromptTemplate(
        template_id="regression_to_mean_trap",
        failure_target=FailureType.PATTERN_MISAPPLICATION,
        template_text=(
            "Observation: {extreme_observation}\n"
            "Intervention: {intervention}\n"
            "Follow-up: {followup_result}\n\n"
            "Question: Did the intervention cause the follow-up result?"
        ),
        parameters=[
            "extreme_observation",
            "intervention",
            "followup_result",
        ],
        description="Mistakes regression to the mean for a causal effect",
    ),
    PromptTemplate(
        template_id="conditional_probability_trap",
        failure_target=FailureType.UNJUSTIFIED_ASSUMPTION,
        template_text=(
            "Given: {conditional_info}\n\n"
            "Question: What is the probability that {inverse_question}?\n"
            "Show your calculation."
        ),
        parameters=["conditional_info", "inverse_question"],
        description="Confuses P(A|B) with P(B|A)",
    ),
    PromptTemplate(
        template_id="vacuous_truth_trap",
        failure_target=FailureType.CONTRADICTION,
        template_text=(
            "Rule: If {false_antecedent}, then {consequent}.\n\n"
            "The antecedent is known to be false.\n"
            "Question: Is the rule violated? Explain."
        ),
        parameters=["false_antecedent", "consequent"],
        description="Mishandles conditionals with false antecedents",
    ),
    PromptTemplate(
        template_id="infinite_regress_trap",
        failure_target=FailureType.MULTI_STEP_BREAK,
        template_text=(
            "Definition: {recursive_definition}\n\n"
            "Question: Using this definition, compute {query}.\n"
            "Show each step."
        ),
        parameters=["recursive_definition", "query"],
        description="Fails to detect non-terminating recursive definitions",
    ),
    PromptTemplate(
        template_id="equivocation_trap",
        failure_target=FailureType.AMBIGUITY_FAILURE,
        template_text=(
            "Premise 1: {premise_with_term}\n"
            "Premise 2: {premise_different_meaning}\n\n"
            "Conclusion: {conclusion}\n"
            "Is this conclusion valid? Why or why not?"
        ),
        parameters=[
            "premise_with_term",
            "premise_different_meaning",
            "conclusion",
        ],
        description="Fails to notice same word used with different meanings",
    ),
)

DISTRIBUTION: dict[str, int] = {
    # Original 10 templates
    "implicit_assumption_trap": 8,
    "contradictory_constraints": 8,
    "edge_case_inversion": 5,
    "false_analogy_trap": 5,
    "multi_step_dependency": 5,
    "recursive_definition_break": 5,
    "ambiguous_spec_trap": 5,
    "overconstrained_optimization": 5,
    "hidden_variable_trap": 3,
    "self_consistency_trap": 3,
    # New 15 templates
    "temporal_ordering_trap": 3,
    "negation_scope_trap": 3,
    "base_rate_neglect": 4,
    "survivorship_bias_trap": 3,
    "modus_tollens_break": 3,
    "scope_creep_trap": 3,
    "anchoring_trap": 4,
    "false_dichotomy_trap": 3,
    "composition_fallacy": 3,
    "conjunction_fallacy": 3,
    "regression_to_mean_trap": 3,
    "conditional_probability_trap": 4,
    "vacuous_truth_trap": 3,
    "infinite_regress_trap": 3,
    "equivocation_trap": 3,
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

    def list_by_failure_type(self, failure_type: FailureType) -> list[PromptTemplate]:
        """Return templates targeting a specific failure type."""
        return [t for t in self._templates.values() if t.failure_target == failure_type]

    def render(self, template_id: str, **params: str) -> str:
        """Render a template with given parameters.

        Raises KeyError if template not found or required parameter missing.
        """
        template = self.get(template_id)
        missing = set(template.parameters) - set(params.keys())
        if missing:
            raise KeyError(f"Missing parameters: {missing}")
        extra = set(params.keys()) - set(template.parameters)
        if extra:
            raise KeyError(f"Unexpected parameters: {extra}")
        return template.template_text.format(**params)

    def template_ids(self) -> list[str]:
        """Return all template IDs."""
        return list(self._templates.keys())
