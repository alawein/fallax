# ReasonBench Phase 3: Intelligence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add analytics, failure prediction, and reasoning clustering on top of Phase 2's JSONL results — enabling data-driven insights about model weaknesses and prompt effectiveness.

**Architecture:** Analyzer reads stored EvaluationResults and computes metrics (accuracy by model, failure rate by type, disagreement rate, assumption density). FailurePredictor trains TF-IDF + logistic regression on prompt texts to predict failure probability before expensive LLM evaluation. FailureClusterer groups reasoning traces with TF-IDF + k-means to surface failure patterns. CLI gains subcommands: `run` (existing pipeline), `analyze`, `train`. All new components consume `list[EvaluationResult]` from `JsonlStore`.

**Tech Stack:** Python 3.12+, scikit-learn (TF-IDF, LogReg, KMeans), Pydantic v2, pytest, uv

**Depends on:** Phase 2 (pipeline, storage, models) — all stable with 160 passing tests.

**Security note:** The predictor uses pickle for scikit-learn model serialization. This is standard practice for ML model persistence. Models are saved/loaded locally by the user — never from untrusted sources.

---

## File Structure

```
reasonbench/
├── reasonbench/
│   ├── __init__.py           # MODIFY: add Analyzer, FailurePredictor, FailureClusterer
│   ├── analyzer.py           # NEW: metrics computation + hard case extraction
│   ├── predictor.py          # NEW: TF-IDF + LogReg failure predictor
│   ├── clusterer.py          # NEW: TF-IDF + k-means reasoning clusterer
│   └── __main__.py           # MODIFY: restructure to subcommands (run/analyze/train)
├── tests/
│   ├── conftest.py           # MODIFY: add make_result helper
│   ├── test_analyzer.py      # NEW
│   ├── test_predictor.py     # NEW
│   ├── test_clusterer.py     # NEW
│   └── test_cli.py           # MODIFY: update for subcommands
```

**Dependency order:** `conftest update` → `analyzer` → (`predictor`, `clusterer`) → `CLI restructure` → `__init__.py`

---

## Task 1: Add scikit-learn Dependency and Test Helpers

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add scikit-learn dependency**

In `pyproject.toml`, change:
```toml
dependencies = [
    "pydantic>=2.0,<3",
    "anthropic>=0.40",
]
```
to:
```toml
dependencies = [
    "pydantic>=2.0,<3",
    "anthropic>=0.40",
    "scikit-learn>=1.5",
]
```

- [ ] **Step 2: Install updated dependencies**

```bash
uv sync --dev
```

- [ ] **Step 3: Add make_result helper to conftest.py**

Append to the end of `tests/conftest.py`:

```python
from reasonbench.models import (
    Assumption,
    EvaluationResult,
    ModelResponse,
    ValidationResult,
)
from reasonbench.taxonomy import FailureType, Severity


def make_result(
    prompt_id: str = "test",
    prompt_text: str = "test prompt",
    failure_type: FailureType = FailureType.CONTRADICTION,
    score: int = 5,
    reasoning_flawed: bool = True,
    answer: str = "42",
    reasoning: str = "Step 1: assume. Step 2: conclude.",
    is_correct: bool = False,
    model_name: str = "test-model",
    assumptions: list[Assumption] | None = None,
) -> EvaluationResult:
    """Build an EvaluationResult for testing."""
    from reasonbench.scoring import Scorer

    severity = Scorer.severity(score)
    return EvaluationResult(
        prompt_id=prompt_id,
        failure_type=failure_type,
        prompt_text=prompt_text,
        models={
            model_name: ModelResponse(
                model_name=model_name,
                answer=answer,
                reasoning=reasoning,
                is_correct=is_correct,
            ),
        },
        validation=ValidationResult(
            reasoning_flawed=reasoning_flawed,
            first_error_step=2 if reasoning_flawed else None,
            assumptions=assumptions or [],
            counterfactual_fail=reasoning_flawed,
            final_answer_correct=not reasoning_flawed,
        ),
        score=score,
        severity=severity,
    )
```

- [ ] **Step 4: Verify existing tests still pass**

```bash
uv run pytest -q
```
Expected: 160 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py
git commit -m "chore: add scikit-learn dependency and make_result test helper"
```

---

## Task 2: Analyzer

**Files:**
- Create: `reasonbench/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analyzer.py`:

```python
import pytest

from reasonbench.analyzer import Analyzer
from reasonbench.models import Assumption
from reasonbench.taxonomy import FailureType, Severity
from tests.conftest import make_result


@pytest.fixture()
def results():
    """Mixed set of 6 results for analytics testing."""
    return [
        make_result(
            prompt_id="1", score=8, reasoning_flawed=True,
            failure_type=FailureType.CONTRADICTION,
            prompt_text="contradiction prompt",
            assumptions=[Assumption(text="a", justified=False)],
        ),
        make_result(
            prompt_id="2", score=7, reasoning_flawed=True,
            failure_type=FailureType.CONTRADICTION,
            prompt_text="another contradiction",
            assumptions=[
                Assumption(text="b", justified=False),
                Assumption(text="c", justified=True),
            ],
        ),
        make_result(
            prompt_id="3", score=3, reasoning_flawed=True,
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            prompt_text="assumption prompt",
        ),
        make_result(
            prompt_id="4", score=1, reasoning_flawed=False,
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            prompt_text="easy assumption",
        ),
        make_result(
            prompt_id="5", score=5, reasoning_flawed=True,
            failure_type=FailureType.OVERGENERALIZATION,
            prompt_text="overgeneralization prompt",
            assumptions=[Assumption(text="d", justified=False)],
        ),
        make_result(
            prompt_id="6", score=0, reasoning_flawed=False,
            failure_type=FailureType.OVERGENERALIZATION,
            prompt_text="easy overgeneralization",
        ),
    ]


class TestAnalyzer:
    def test_summary_total(self, results):
        a = Analyzer(results)
        s = a.summary()
        assert s["total"] == 6

    def test_summary_avg_score(self, results):
        a = Analyzer(results)
        s = a.summary()
        # (8+7+3+1+5+0)/6 = 4.0
        assert abs(s["avg_score"] - 4.0) < 0.01

    def test_summary_failure_rate(self, results):
        a = Analyzer(results)
        s = a.summary()
        assert abs(s["failure_rate"] - 4 / 6) < 0.01

    def test_summary_severity_counts(self, results):
        a = Analyzer(results)
        s = a.summary()
        counts = s["severity_counts"]
        assert counts[Severity.CRITICAL] == 2  # scores 8, 7
        assert counts[Severity.HIGH] == 1  # score 5
        assert counts[Severity.LOW] == 2  # scores 1, 0

    def test_failure_rate_by_type(self, results):
        a = Analyzer(results)
        rates = a.failure_rate_by_type()
        assert rates["contradiction"] == 1.0
        assert rates["unstated_assumption"] == 0.5

    def test_model_accuracy(self, results):
        a = Analyzer(results)
        acc = a.model_accuracy()
        assert "test-model" in acc
        assert abs(acc["test-model"] - 2 / 6) < 0.01

    def test_top_failures(self, results):
        a = Analyzer(results)
        top = a.top_failures(n=3)
        assert len(top) == 3
        assert top[0].score >= top[1].score >= top[2].score
        assert top[0].score == 8

    def test_hard_cases(self, results):
        a = Analyzer(results)
        hard = a.hard_cases(min_score=6)
        assert len(hard) == 2  # scores 8, 7
        assert all(r.score >= 6 for r in hard)

    def test_hard_cases_empty(self, results):
        a = Analyzer(results)
        hard = a.hard_cases(min_score=100)
        assert hard == []

    def test_disagreement_rate_single_model(self, results):
        a = Analyzer(results)
        assert a.disagreement_rate() == 0.0

    def test_assumption_density(self, results):
        a = Analyzer(results)
        density = a.assumption_density()
        assert abs(density - 3 / 6) < 0.01

    def test_empty_results(self):
        a = Analyzer([])
        s = a.summary()
        assert s["total"] == 0
        assert s["avg_score"] == 0.0
        assert s["failure_rate"] == 0.0
        assert a.disagreement_rate() == 0.0
        assert a.assumption_density() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/analyzer.py`:

```python
"""Analytics engine for evaluation results."""

from __future__ import annotations

from collections import Counter, defaultdict

from .models import EvaluationResult
from .taxonomy import Severity


class Analyzer:
    """Computes metrics and extracts insights from evaluation results."""

    def __init__(self, results: list[EvaluationResult]) -> None:
        self._results = results

    def summary(self) -> dict:
        """High-level summary statistics."""
        n = len(self._results)
        if n == 0:
            return {
                "total": 0,
                "severity_counts": {},
                "avg_score": 0.0,
                "failure_rate": 0.0,
            }
        return {
            "total": n,
            "severity_counts": dict(
                Counter(r.severity for r in self._results)
            ),
            "avg_score": sum(r.score for r in self._results) / n,
            "failure_rate": (
                sum(1 for r in self._results if r.validation.reasoning_flawed)
                / n
            ),
        }

    def failure_rate_by_type(self) -> dict[str, float]:
        """Reasoning failure rate per failure type."""
        by_type: dict[str, list[bool]] = defaultdict(list)
        for r in self._results:
            by_type[r.failure_type.value].append(
                r.validation.reasoning_flawed
            )
        return {
            ft: sum(flags) / len(flags)
            for ft, flags in by_type.items()
        }

    def model_accuracy(self) -> dict[str, float]:
        """Accuracy per model based on truth judge verdict."""
        by_model: dict[str, list[bool]] = defaultdict(list)
        for r in self._results:
            correct = r.validation.final_answer_correct or False
            for model_name in r.models:
                by_model[model_name].append(correct)
        return {
            model: sum(flags) / len(flags)
            for model, flags in by_model.items()
        }

    def top_failures(self, n: int = 10) -> list[EvaluationResult]:
        """Top N highest-scoring failures."""
        return sorted(
            self._results, key=lambda r: r.score, reverse=True
        )[:n]

    def hard_cases(self, min_score: int = 6) -> list[EvaluationResult]:
        """Extract cases with score >= threshold."""
        return [r for r in self._results if r.score >= min_score]

    def disagreement_rate(self) -> float:
        """Rate of model disagreement across results."""
        if not self._results:
            return 0.0
        disagreements = sum(
            1
            for r in self._results
            if len({resp.answer for resp in r.models.values()}) > 1
        )
        return disagreements / len(self._results)

    def assumption_density(self) -> float:
        """Average number of unjustified assumptions per result."""
        if not self._results:
            return 0.0
        total = sum(
            sum(1 for a in r.validation.assumptions if not a.justified)
            for r in self._results
        )
        return total / len(self._results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/analyzer.py tests/test_analyzer.py
git commit -m "feat(analyzer): add metrics computation and hard case extraction"
```

---

## Task 3: Failure Predictor

**Files:**
- Create: `reasonbench/predictor.py`
- Create: `tests/test_predictor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_predictor.py`:

```python
from pathlib import Path

import pytest

from reasonbench.predictor import FailurePredictor
from reasonbench.taxonomy import FailureType
from tests.conftest import make_result


@pytest.fixture()
def training_data():
    """20 results with varied scores and distinct prompt texts."""
    data = []
    high_words = [
        "contradictory constraints impossible",
        "recursive definition undefined",
        "edge case boundary fails",
        "implicit assumption hidden rule",
        "overconstrained no solution exists",
        "ambiguous specification unclear",
        "multi step dependency chain error",
        "false analogy pattern mismatch",
        "self consistency logical flaw",
        "hidden variable missing information",
    ]
    low_words = [
        "simple arithmetic addition",
        "basic string reversal",
        "trivial sorting ascending",
        "easy boolean logic gate",
        "straightforward lookup table",
        "plain concatenation operation",
        "direct mapping function",
        "clear input validation",
        "obvious pattern matching",
        "standard list filtering",
    ]
    for i, text in enumerate(high_words):
        data.append(make_result(
            prompt_id=f"high-{i}",
            prompt_text=text,
            score=7,
            reasoning_flawed=True,
        ))
    for i, text in enumerate(low_words):
        data.append(make_result(
            prompt_id=f"low-{i}",
            prompt_text=text,
            score=1,
            reasoning_flawed=False,
        ))
    return data


class TestFailurePredictor:
    def test_not_fitted_initially(self):
        p = FailurePredictor()
        assert p.is_fitted is False

    def test_predict_before_train_raises(self):
        p = FailurePredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            p.predict("some prompt")

    def test_predict_batch_before_train_raises(self):
        p = FailurePredictor()
        with pytest.raises(RuntimeError, match="not trained"):
            p.predict_batch(["a", "b"])

    def test_train_returns_metrics(self, training_data):
        p = FailurePredictor()
        metrics = p.train(training_data, threshold=4)
        assert "samples" in metrics
        assert metrics["samples"] == 20
        assert "positive_rate" in metrics
        assert abs(metrics["positive_rate"] - 0.5) < 0.01
        assert "cv_accuracy" in metrics
        assert 0.0 <= metrics["cv_accuracy"] <= 1.0

    def test_is_fitted_after_train(self, training_data):
        p = FailurePredictor()
        p.train(training_data)
        assert p.is_fitted is True

    def test_predict_returns_probability(self, training_data):
        p = FailurePredictor()
        p.train(training_data)
        prob = p.predict("contradictory constraints impossible situation")
        assert 0.0 <= prob <= 1.0

    def test_predict_batch_returns_list(self, training_data):
        p = FailurePredictor()
        p.train(training_data)
        probs = p.predict_batch(["hard constraint problem", "easy simple task"])
        assert len(probs) == 2
        assert all(0.0 <= prob <= 1.0 for prob in probs)

    def test_high_failure_prompt_scores_higher(self, training_data):
        p = FailurePredictor()
        p.train(training_data, threshold=4)
        high_prob = p.predict("contradictory recursive undefined edge case")
        low_prob = p.predict("simple basic easy trivial straightforward")
        assert high_prob > low_prob

    def test_save_and_load(self, training_data, tmp_path):
        p = FailurePredictor()
        p.train(training_data)
        model_path = tmp_path / "predictor.pkl"
        p.save(model_path)
        assert model_path.exists()

        p2 = FailurePredictor()
        assert p2.is_fitted is False
        p2.load(model_path)
        assert p2.is_fitted is True

        prob1 = p.predict("test prompt")
        prob2 = p2.predict("test prompt")
        assert abs(prob1 - prob2) < 0.001

    def test_train_with_custom_threshold(self, training_data):
        p = FailurePredictor()
        metrics = p.train(training_data, threshold=2)
        assert metrics["positive_rate"] > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/predictor.py`:

```python
"""Failure predictor using TF-IDF + logistic regression.

Security note: Uses pickle for scikit-learn model serialization.
Only load models you have trained yourself.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .models import EvaluationResult


class FailurePredictor:
    """Predicts failure probability for prompts using TF-IDF features.

    Train on existing evaluation results, then predict failure probability
    for new prompts to filter low-value evaluations before expensive LLM runs.
    Prompts below 0.3 probability can be skipped.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english"
        )
        self._classifier = LogisticRegression(max_iter=1000)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the predictor has been trained."""
        return self._fitted

    def train(
        self, results: list[EvaluationResult], threshold: int = 4
    ) -> dict:
        """Train on results. Labels: score >= threshold = failure.

        Returns training metrics including cross-validation accuracy.
        """
        texts = [r.prompt_text for r in results]
        labels = [1 if r.score >= threshold else 0 for r in results]

        X = self._vectorizer.fit_transform(texts)
        self._classifier.fit(X, labels)
        self._fitted = True

        cv_folds = min(5, len(results))
        scores = cross_val_score(
            self._classifier, X, labels, cv=cv_folds
        )
        return {
            "samples": len(results),
            "positive_rate": sum(labels) / max(1, len(labels)),
            "cv_accuracy": float(scores.mean()),
        }

    def predict(self, prompt_text: str) -> float:
        """Predict failure probability for a single prompt."""
        if not self._fitted:
            raise RuntimeError("Predictor not trained. Call train() first.")
        X = self._vectorizer.transform([prompt_text])
        return float(self._classifier.predict_proba(X)[0, 1])

    def predict_batch(self, texts: list[str]) -> list[float]:
        """Predict failure probabilities for multiple prompts."""
        if not self._fitted:
            raise RuntimeError("Predictor not trained. Call train() first.")
        X = self._vectorizer.transform(texts)
        return [float(p) for p in self._classifier.predict_proba(X)[:, 1]]

    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "classifier": self._classifier,
                },
                f,
            )

    def load(self, path: Path) -> None:
        """Load trained model from disk. Only load models you trust."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._vectorizer = data["vectorizer"]
        self._classifier = data["classifier"]
        self._fitted = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_predictor.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/predictor.py tests/test_predictor.py
git commit -m "feat(predictor): add TF-IDF + logistic regression failure predictor"
```

---

## Task 4: Failure Clusterer

**Files:**
- Create: `reasonbench/clusterer.py`
- Create: `tests/test_clusterer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_clusterer.py`:

```python
import pytest

from reasonbench.clusterer import FailureClusterer
from reasonbench.taxonomy import FailureType
from tests.conftest import make_result


@pytest.fixture()
def cluster_data():
    """15 results with distinct reasoning patterns for clustering."""
    data = []
    for i in range(5):
        data.append(make_result(
            prompt_id=f"math-{i}",
            prompt_text=f"math problem {i}",
            reasoning=f"Step 1: apply formula. Step 2: compute {i}. Step 3: derive answer.",
            failure_type=FailureType.CONTRADICTION,
            score=6,
        ))
    for i in range(5):
        data.append(make_result(
            prompt_id=f"assume-{i}",
            prompt_text=f"assumption problem {i}",
            reasoning=f"Step 1: assume input valid. Step 2: assume positive. Step 3: conclude {i}.",
            failure_type=FailureType.UNSTATED_ASSUMPTION,
            score=5,
        ))
    for i in range(5):
        data.append(make_result(
            prompt_id=f"constraint-{i}",
            prompt_text=f"constraint problem {i}",
            reasoning=f"Step 1: check constraint. Step 2: verify bounds. Step 3: satisfy {i}.",
            failure_type=FailureType.IGNORED_CONSTRAINT,
            score=7,
        ))
    return data


class TestFailureClusterer:
    def test_fit_assigns_labels(self, cluster_data):
        c = FailureClusterer(n_clusters=3)
        c.fit(cluster_data)
        assert len(c.labels) == 15
        assert set(c.labels) == {0, 1, 2}

    def test_cluster_summary_keys(self, cluster_data):
        c = FailureClusterer(n_clusters=3)
        c.fit(cluster_data)
        summary = c.cluster_summary()
        assert len(summary) == 3
        for label, info in summary.items():
            assert "size" in info
            assert "avg_score" in info
            assert "dominant_failure_type" in info
            assert "failure_type_distribution" in info

    def test_cluster_sizes_sum_to_total(self, cluster_data):
        c = FailureClusterer(n_clusters=3)
        c.fit(cluster_data)
        summary = c.cluster_summary()
        total = sum(info["size"] for info in summary.values())
        assert total == 15

    def test_cluster_avg_scores_reasonable(self, cluster_data):
        c = FailureClusterer(n_clusters=3)
        c.fit(cluster_data)
        summary = c.cluster_summary()
        for info in summary.values():
            assert 0 <= info["avg_score"] <= 10

    def test_labels_empty_before_fit(self):
        c = FailureClusterer(n_clusters=3)
        assert c.labels == []

    def test_two_clusters(self, cluster_data):
        c = FailureClusterer(n_clusters=2)
        c.fit(cluster_data)
        assert len(c.labels) == 15
        assert set(c.labels) == {0, 1}

    def test_single_cluster(self, cluster_data):
        c = FailureClusterer(n_clusters=1)
        c.fit(cluster_data)
        assert all(label == 0 for label in c.labels)
        summary = c.cluster_summary()
        assert summary[0]["size"] == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_clusterer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

Create `reasonbench/clusterer.py`:

```python
"""Failure clustering using TF-IDF + k-means on reasoning traces."""

from __future__ import annotations

from collections import Counter, defaultdict

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import EvaluationResult


class FailureClusterer:
    """Groups reasoning traces to surface failure patterns.

    Uses TF-IDF to vectorize model reasoning, then k-means to cluster.
    Cluster summaries reveal which failure types and score ranges
    co-occur, exposing model blind spots.
    """

    def __init__(self, n_clusters: int = 5) -> None:
        self._n_clusters = n_clusters
        self._vectorizer = TfidfVectorizer(
            max_features=3000, stop_words="english"
        )
        self._kmeans = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        )
        self._labels: list[int] = []
        self._results: list[EvaluationResult] = []

    @property
    def labels(self) -> list[int]:
        """Cluster label for each result (empty before fit)."""
        return self._labels

    def fit(self, results: list[EvaluationResult]) -> None:
        """Cluster reasoning traces from results."""
        self._results = results
        texts = [
            next(iter(r.models.values())).reasoning for r in results
        ]
        X = self._vectorizer.fit_transform(texts)
        self._kmeans.fit(X)
        self._labels = [int(label) for label in self._kmeans.labels_]

    def cluster_summary(self) -> dict[int, dict]:
        """Summary statistics per cluster."""
        clusters: dict[int, list[EvaluationResult]] = defaultdict(list)
        for i, label in enumerate(self._labels):
            clusters[label].append(self._results[i])

        summaries: dict[int, dict] = {}
        for label, cluster_results in sorted(clusters.items()):
            failure_types = Counter(
                r.failure_type.value for r in cluster_results
            )
            summaries[label] = {
                "size": len(cluster_results),
                "avg_score": (
                    sum(r.score for r in cluster_results)
                    / len(cluster_results)
                ),
                "dominant_failure_type": failure_types.most_common(1)[0][0],
                "failure_type_distribution": dict(failure_types),
            }
        return summaries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_clusterer.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/clusterer.py tests/test_clusterer.py
git commit -m "feat(clusterer): add TF-IDF + k-means reasoning trace clustering"
```

---

## Task 5: CLI Restructure with Subcommands

**Files:**
- Modify: `reasonbench/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write updated CLI tests**

Replace the content of `tests/test_cli.py` with:

```python
"""Tests for CLI subcommands."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reasonbench.__main__ import main
from tests.conftest import (
    JUDGE_RESPONSES,
    MODEL_RESPONSE_TEXT,
    MockClient,
    make_result,
)


@pytest.fixture()
def params_dir(tmp_path: Path) -> Path:
    data = [
        {"rule_a": "x > 0", "rule_b": "double", "edge_case_input": "x = -1"},
    ]
    (tmp_path / "implicit_assumption_trap.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture()
def results_file(tmp_path: Path) -> Path:
    """Write mock results to a JSONL file."""
    path = tmp_path / "results.jsonl"
    results = [
        make_result(prompt_id=f"r{i}", score=i * 2, reasoning_flawed=i > 2)
        for i in range(6)
    ]
    with open(path, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    return path


class TestRunSubcommand:
    def test_run_returns_zero(self, params_dir, tmp_path):
        output = tmp_path / "out.jsonl"
        mock = MockClient(
            responses=JUDGE_RESPONSES, default=MODEL_RESPONSE_TEXT
        )
        with patch("reasonbench.__main__.AnthropicClient", return_value=mock):
            code = main([
                "run",
                "--models", "m",
                "--judge", "j",
                "--count", "1",
                "--output", str(output),
                "--params-dir", str(params_dir),
                "--seed", "42",
            ])
        assert code == 0
        assert output.exists()


class TestAnalyzeSubcommand:
    def test_analyze_returns_zero(self, results_file):
        code = main(["analyze", str(results_file)])
        assert code == 0

    def test_analyze_with_top_flag(self, results_file):
        code = main(["analyze", str(results_file), "--top", "3"])
        assert code == 0

    def test_analyze_missing_file_returns_error(self, tmp_path):
        code = main(["analyze", str(tmp_path / "nonexistent.jsonl")])
        assert code == 1


class TestTrainSubcommand:
    def test_train_returns_zero(self, results_file, tmp_path):
        model_path = tmp_path / "predictor.pkl"
        code = main([
            "train", str(results_file),
            "--output", str(model_path),
        ])
        assert code == 0
        assert model_path.exists()

    def test_train_missing_file_returns_error(self, tmp_path):
        code = main([
            "train", str(tmp_path / "nonexistent.jsonl"),
            "--output", str(tmp_path / "model.pkl"),
        ])
        assert code == 1


class TestNoSubcommand:
    def test_no_args_returns_one(self):
        code = main([])
        assert code == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (old CLI doesn't have subcommands)

- [ ] **Step 3: Rewrite __main__.py with subcommands**

Read existing `reasonbench/__main__.py`, then replace entirely with:

```python
"""CLI entry point: python -m reasonbench."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .client import AnthropicClient
from .pipeline import Pipeline


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the evaluation pipeline."""
    client = AnthropicClient()
    pipeline = Pipeline(
        client=client,
        models=args.models,
        judge_model=args.judge,
        output_path=Path(args.output),
        params_dir=Path(args.params_dir) if args.params_dir else None,
        seed=args.seed,
    )
    results = pipeline.run(count=args.count)

    critical = sum(1 for r in results if r.severity.value == "critical")
    high = sum(1 for r in results if r.severity.value == "high")
    medium = sum(1 for r in results if r.severity.value == "medium")
    low = sum(1 for r in results if r.severity.value == "low")

    print(f"\nReasonBench Evaluation Complete")
    print(f"  Prompts evaluated: {len(results)}")
    print(f"  Critical: {critical}")
    print(f"  High:     {high}")
    print(f"  Medium:   {medium}")
    print(f"  Low:      {low}")
    print(f"  Output:   {args.output}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze evaluation results."""
    from .analyzer import Analyzer
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results found.")
        return 0

    analyzer = Analyzer(results)
    summary = analyzer.summary()

    print(f"\nReasonBench Analysis ({summary['total']} results)")
    print(f"  Avg score:     {summary['avg_score']:.2f}")
    print(f"  Failure rate:  {summary['failure_rate']:.1%}")
    print(f"  Disagreement:  {analyzer.disagreement_rate():.1%}")
    print(f"  Assumption density: {analyzer.assumption_density():.2f}")

    print(f"\nSeverity distribution:")
    for sev, count in sorted(
        summary["severity_counts"].items(), key=lambda x: x[0].value
    ):
        print(f"  {sev.value:>8}: {count}")

    print(f"\nFailure rate by type:")
    for ft, rate in sorted(
        analyzer.failure_rate_by_type().items(), key=lambda x: -x[1]
    ):
        print(f"  {ft:>30}: {rate:.1%}")

    print(f"\nModel accuracy:")
    for model, acc in analyzer.model_accuracy().items():
        print(f"  {model}: {acc:.1%}")

    top = analyzer.top_failures(n=args.top)
    if top:
        print(f"\nTop {len(top)} failures:")
        for r in top:
            print(
                f"  [{r.severity.value:>8}] score={r.score} "
                f"type={r.failure_type.value} id={r.prompt_id}"
            )

    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train failure predictor on results."""
    from .predictor import FailurePredictor
    from .storage import JsonlStore

    path = Path(args.results)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    store = JsonlStore(path)
    results = store.read_all()
    if not results:
        print("No results to train on.")
        return 1

    predictor = FailurePredictor()
    metrics = predictor.train(results, threshold=args.threshold)

    output = Path(args.output)
    predictor.save(output)

    print(f"\nPredictor trained")
    print(f"  Samples:     {metrics['samples']}")
    print(f"  Positive rate: {metrics['positive_rate']:.1%}")
    print(f"  CV accuracy:   {metrics['cv_accuracy']:.1%}")
    print(f"  Saved to:    {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """ReasonBench CLI entry point."""
    default_model = os.environ.get("REASONBENCH_MODEL", "")
    default_judge = os.environ.get("REASONBENCH_JUDGE_MODEL", "")

    parser = argparse.ArgumentParser(
        prog="reasonbench",
        description="LLM Adversarial Reasoning Evaluation System",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- run --
    run_p = subparsers.add_parser("run", help="Run evaluation pipeline")
    run_p.add_argument(
        "--models", nargs="+",
        required=not bool(default_model),
        default=[default_model] if default_model else None,
        help="Models to evaluate (or set REASONBENCH_MODEL env var)",
    )
    run_p.add_argument(
        "--judge",
        required=not bool(default_judge),
        default=default_judge or None,
        help="Judge model (or set REASONBENCH_JUDGE_MODEL env var)",
    )
    run_p.add_argument("--count", type=int, default=10)
    run_p.add_argument("--output", default="results.jsonl")
    run_p.add_argument("--params-dir", default=None)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--verbose", "-v", action="store_true")

    # -- analyze --
    analyze_p = subparsers.add_parser(
        "analyze", help="Analyze evaluation results"
    )
    analyze_p.add_argument("results", help="Path to results JSONL file")
    analyze_p.add_argument(
        "--top", type=int, default=10, help="Number of top failures to show"
    )

    # -- train --
    train_p = subparsers.add_parser(
        "train", help="Train failure predictor"
    )
    train_p.add_argument("results", help="Path to results JSONL file")
    train_p.add_argument(
        "--output", "-o", default="predictor.pkl",
        help="Output model file (default: predictor.pkl)",
    )
    train_p.add_argument(
        "--threshold", type=int, default=4,
        help="Score threshold for failure label (default: 4)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO
        if getattr(args, "verbose", False)
        else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "train":
        return _cmd_train(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add reasonbench/__main__.py tests/test_cli.py
git commit -m "feat(cli): restructure to subcommands (run/analyze/train)"
```

---

## Task 6: Public API Update and Full Suite Verification

**Files:**
- Modify: `reasonbench/__init__.py`

- [ ] **Step 1: Update __init__.py**

Read existing `reasonbench/__init__.py`, then replace with:

```python
"""ReasonBench — LLM Adversarial Reasoning Evaluation System."""

from .analyzer import Analyzer
from .client import AnthropicClient, LLMClient
from .clusterer import FailureClusterer
from .evaluator import Evaluator
from .generator import PromptGenerator
from .models import (
    Assumption,
    EvaluationResult,
    FailureRecord,
    ModelResponse,
    Prompt,
    ValidationResult,
)
from .pipeline import Pipeline
from .predictor import FailurePredictor
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
    "PromptGenerator",
    "Scorer",
    "Severity",
    "TemplateRegistry",
    "ValidationResult",
    "ValidatorPack",
    "get_category",
]
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest -v
```
Expected: All tests PASS (~200 tests)

- [ ] **Step 3: Commit**

```bash
git add reasonbench/__init__.py
git commit -m "feat: update public API with Phase 3 exports (Analyzer, Predictor, Clusterer)"
```

---

## Summary

| Task | Component | New/Modified Files | Tests |
|------|-----------|-------------------|-------|
| 1 | Dependencies + Helpers | 2 (pyproject, conftest) | 0 |
| 2 | Analyzer | 2 (analyzer, test) | 13 |
| 3 | Failure Predictor | 2 (predictor, test) | 10 |
| 4 | Failure Clusterer | 2 (clusterer, test) | 7 |
| 5 | CLI Subcommands | 2 (__main__, test_cli) | 7 |
| 6 | Init Update | 1 (__init__) | 0 |
| **Total** | | **11 files** | **~37 new tests** |

**Phase 3 delivers:** Analytics engine computing accuracy/failure rates per model and type, disagreement rates, assumption density. Hard case extraction for prompt evolution feed. TF-IDF + logistic regression predictor that filters low-value prompts before expensive LLM evaluation. K-means clustering of reasoning traces to surface model blind spots. CLI subcommands: `reasonbench run`, `reasonbench analyze results.jsonl`, `reasonbench train results.jsonl -o predictor.pkl`.
