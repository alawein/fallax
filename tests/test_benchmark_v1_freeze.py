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
    # Normalize CRLF -> LF before hashing so the test is stable on Windows
    # working trees that Dropbox or an editor flipped, even though
    # .gitattributes guarantees the index itself is LF.
    raw = (V1_DIR / "prompts.jsonl").read_bytes().replace(b"\r\n", b"\n")
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


def test_template_count_matches_metadata(metadata: dict, prompts: list[dict]) -> None:
    """Metadata claims 25 templates; the prompts must actually use 25 distinct ones.

    Without this, a metadata-only edit could silently change the claimed
    template count while the prompts file still uses the old set.
    """
    declared = metadata["generation_params"]["templates"]
    distinct = {p["template_id"] for p in prompts}
    assert len(distinct) == declared == 25, (
        f"metadata declares {declared} templates; prompts use {len(distinct)} distinct"
    )


def test_no_single_failure_type_dominates(prompts: list[dict]) -> None:
    """Distribution is metadata-declared as 'weighted'; cap at 25% to guard against collapse.

    25% is roughly 2.5x the uniform share (10%) over 10 failure_types. A
    future regeneration that collapses to a near-single-type set would
    invalidate per-category comparability claims; this test catches that.
    """
    from collections import Counter

    counts = Counter(p["failure_type"] for p in prompts)
    top_type, top_count = counts.most_common(1)[0]
    share = top_count / len(prompts)
    assert share <= 0.25, (
        f"failure_type {top_type!r} dominates v1: {top_count}/{len(prompts)} = {share:.0%}"
    )
