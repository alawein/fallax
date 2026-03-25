"""Validate parameter bank JSON files match template definitions."""

import json
from pathlib import Path

import pytest

from reasonbench.templates import TEMPLATES, TemplateRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "reasonbench" / "data"


class TestParameterBanks:
    def test_all_templates_have_data_files(self):
        for t in TEMPLATES:
            path = DATA_DIR / f"{t.template_id}.json"
            assert path.exists(), f"Missing data file for {t.template_id}"

    @pytest.mark.parametrize(
        "template",
        TEMPLATES,
        ids=[t.template_id for t in TEMPLATES],
    )
    def test_data_file_is_valid_json_array(self, template):
        path = DATA_DIR / f"{template.template_id}.json"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.parametrize(
        "template",
        TEMPLATES,
        ids=[t.template_id for t in TEMPLATES],
    )
    def test_parameter_keys_match_template(self, template):
        path = DATA_DIR / f"{template.template_id}.json"
        with open(path) as f:
            data = json.load(f)
        expected_keys = set(template.parameters)
        for i, entry in enumerate(data):
            assert set(entry.keys()) == expected_keys, (
                f"{template.template_id}[{i}]: "
                f"expected keys {expected_keys}, got {set(entry.keys())}"
            )

    @pytest.mark.parametrize(
        "template",
        TEMPLATES,
        ids=[t.template_id for t in TEMPLATES],
    )
    def test_parameters_render_without_error(self, template):
        path = DATA_DIR / f"{template.template_id}.json"
        with open(path) as f:
            data = json.load(f)
        registry = TemplateRegistry()
        for entry in data:
            rendered = registry.render(template.template_id, **entry)
            assert len(rendered) > 0
