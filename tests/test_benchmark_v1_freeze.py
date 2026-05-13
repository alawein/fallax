"""Freeze test for benchmarks/v1.

Asserts the v1 prompt set is byte-stable, fully populated, and that every
declared failure type and category appears at least once. Any hand-edit of
prompts.jsonl or metadata.json that drifts the SHA-256 or taxonomy coverage
fails this test, preserving cross-model comparability guarantees.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

V1_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "v1"


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((V1_DIR / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prompts() -> list[dict]:
    raw = (V1_DIR / "prompts.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_prompt_count_matches_metadata(metadata: dict, prompts: list[dict]) -> None:
    assert len(prompts) == metadata["prompt_count"] == 100


def test_prompts_sha256_matches_metadata(metadata: dict) -> None:
    raw = (V1_DIR / "prompts.jsonl").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == metadata["prompts_sha256"], (
        "prompts.jsonl changed; if intentional, bump the benchmark version "
        "rather than mutating v1 in place."
    )


def test_every_failure_type_appears(metadata: dict, prompts: list[dict]) -> None:
    seen = {p.get("failure_type") for p in prompts}
    missing = set(metadata["failure_types"]) - seen
    assert not missing, f"failure_types missing from v1 prompts: {missing}"


def test_every_category_appears(metadata: dict, prompts: list[dict]) -> None:
    from fallax.taxonomy import FailureType, get_category

    seen: set[str] = set()
    for p in prompts:
        ft = FailureType(p["failure_type"])
        seen.add(get_category(ft).value)
    missing = set(metadata["categories"]) - seen
    assert not missing, f"categories missing from v1 prompts: {missing}"
