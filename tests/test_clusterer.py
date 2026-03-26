import pytest

from reasonbench.clusterer import FailureClusterer
from reasonbench.taxonomy import FailureType
from tests.conftest import make_result


@pytest.fixture()
def cluster_data():
    data = []
    for i in range(5):
        data.append(
            make_result(
                prompt_id=f"math-{i}",
                prompt_text=f"math problem {i}",
                reasoning=f"Step 1: apply formula. Step 2: compute {i}. Step 3: derive answer.",
                failure_type=FailureType.CONTRADICTION,
                score=6,
            )
        )
    for i in range(5):
        data.append(
            make_result(
                prompt_id=f"assume-{i}",
                prompt_text=f"assumption problem {i}",
                reasoning=f"Step 1: assume input valid. Step 2: assume positive. Step 3: conclude {i}.",
                failure_type=FailureType.UNSTATED_ASSUMPTION,
                score=5,
            )
        )
    for i in range(5):
        data.append(
            make_result(
                prompt_id=f"constraint-{i}",
                prompt_text=f"constraint problem {i}",
                reasoning=f"Step 1: check constraint. Step 2: verify bounds. Step 3: satisfy {i}.",
                failure_type=FailureType.IGNORED_CONSTRAINT,
                score=7,
            )
        )
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
        for _label, info in summary.items():
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
