import pytest

from reasonbench.taxonomy import FailureType
from reasonbench.templates import (
    DISTRIBUTION,
    TEMPLATES,
    TemplateRegistry,
)


class TestPromptTemplate:
    def test_is_frozen_dataclass(self):
        t = TEMPLATES[0]
        with pytest.raises(AttributeError):
            t.template_id = "changed"


class TestTemplateConstants:
    def test_ten_templates_defined(self):
        assert len(TEMPLATES) == 10

    def test_all_have_unique_ids(self):
        ids = [t.template_id for t in TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_all_have_non_empty_parameters(self):
        for t in TEMPLATES:
            assert len(t.parameters) > 0, f"{t.template_id} has no parameters"

    def test_all_template_texts_contain_placeholders(self):
        for t in TEMPLATES:
            for param in t.parameters:
                assert f"{{{param}}}" in t.template_text, (
                    f"{t.template_id} missing placeholder {{{param}}}"
                )

    def test_distribution_sums_to_100(self):
        assert sum(DISTRIBUTION.values()) == 100

    def test_distribution_keys_match_template_ids(self):
        template_ids = {t.template_id for t in TEMPLATES}
        assert set(DISTRIBUTION.keys()) == template_ids


class TestTemplateRegistry:
    @pytest.fixture()
    def registry(self):
        return TemplateRegistry()

    def test_get_existing(self, registry):
        t = registry.get("implicit_assumption_trap")
        assert t.template_id == "implicit_assumption_trap"

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_all(self, registry):
        all_templates = registry.list_all()
        assert len(all_templates) == 10

    def test_list_by_failure_type(self, registry):
        results = registry.list_by_failure_type(FailureType.UNSTATED_ASSUMPTION)
        assert len(results) >= 1
        assert all(t.failure_target == FailureType.UNSTATED_ASSUMPTION for t in results)

    def test_render_success(self, registry):
        rendered = registry.render(
            "implicit_assumption_trap",
            rule_a="x > 0",
            rule_b="x < 10",
            edge_case_input="x = 0",
        )
        assert "x > 0" in rendered
        assert "x < 10" in rendered
        assert "x = 0" in rendered

    def test_render_missing_param_raises(self, registry):
        with pytest.raises(KeyError, match="Missing parameters"):
            registry.render("implicit_assumption_trap", rule_a="x > 0")

    def test_render_extra_param_raises(self, registry):
        with pytest.raises(KeyError, match="Unexpected parameters"):
            registry.render(
                "implicit_assumption_trap",
                rule_a="x > 0",
                rule_b="x < 10",
                edge_case_input="x = 0",
                typo_param="oops",
            )

    def test_template_ids(self, registry):
        ids = registry.template_ids()
        assert len(ids) == 10
        assert "implicit_assumption_trap" in ids
