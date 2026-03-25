from reasonbench.taxonomy import (
    FailureCategory,
    FailureType,
    Severity,
    FAILURE_CATEGORY_MAP,
    get_category,
)


class TestFailureCategory:
    def test_has_six_categories(self):
        assert len(FailureCategory) == 6

    def test_values(self):
        expected = {
            "logic_error",
            "assumption_error",
            "constraint_violation",
            "generalization_error",
            "ambiguity_failure",
            "multi_step_break",
        }
        assert {c.value for c in FailureCategory} == expected


class TestFailureType:
    def test_has_ten_types(self):
        assert len(FailureType) == 10

    def test_values(self):
        expected = {
            "contradiction",
            "invalid_inference",
            "unstated_assumption",
            "unjustified_assumption",
            "ignored_constraint",
            "partial_satisfaction",
            "overgeneralization",
            "pattern_misapplication",
            "ambiguity_failure",
            "multi_step_break",
        }
        assert {t.value for t in FailureType} == expected


class TestSeverity:
    def test_has_four_levels(self):
        assert len(Severity) == 4

    def test_ordering(self):
        levels = [s.value for s in Severity]
        assert levels == ["critical", "high", "medium", "low"]


class TestCategoryMap:
    def test_covers_all_failure_types(self):
        assert set(FAILURE_CATEGORY_MAP.keys()) == set(FailureType)

    def test_logic_error_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.CONTRADICTION] == FailureCategory.LOGIC_ERROR
        assert FAILURE_CATEGORY_MAP[FailureType.INVALID_INFERENCE] == FailureCategory.LOGIC_ERROR

    def test_assumption_error_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.UNSTATED_ASSUMPTION] == FailureCategory.ASSUMPTION_ERROR
        assert FAILURE_CATEGORY_MAP[FailureType.UNJUSTIFIED_ASSUMPTION] == FailureCategory.ASSUMPTION_ERROR

    def test_constraint_violation_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.IGNORED_CONSTRAINT] == FailureCategory.CONSTRAINT_VIOLATION
        assert FAILURE_CATEGORY_MAP[FailureType.PARTIAL_SATISFACTION] == FailureCategory.CONSTRAINT_VIOLATION

    def test_generalization_error_subtypes(self):
        assert FAILURE_CATEGORY_MAP[FailureType.OVERGENERALIZATION] == FailureCategory.GENERALIZATION_ERROR
        assert FAILURE_CATEGORY_MAP[FailureType.PATTERN_MISAPPLICATION] == FailureCategory.GENERALIZATION_ERROR


class TestGetCategory:
    def test_returns_correct_category(self):
        assert get_category(FailureType.CONTRADICTION) == FailureCategory.LOGIC_ERROR

    def test_every_type_resolves(self):
        for ft in FailureType:
            result = get_category(ft)
            assert isinstance(result, FailureCategory)
