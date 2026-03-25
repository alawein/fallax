"""Failure clustering using TF-IDF + k-means on reasoning traces."""

from __future__ import annotations

from collections import Counter, defaultdict

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import EvaluationResult


class FailureClusterer:
    """Groups reasoning traces to surface failure patterns."""

    def __init__(self, n_clusters: int = 5) -> None:
        self._n_clusters = n_clusters
        self._vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self._labels: list[int] = []
        self._results: list[EvaluationResult] = []

    @property
    def labels(self) -> list[int]:
        return self._labels

    def fit(self, results: list[EvaluationResult]) -> None:
        self._results = results
        texts = [next(iter(r.models.values())).reasoning for r in results]
        X = self._vectorizer.fit_transform(texts)
        self._kmeans.fit(X)
        self._labels = [int(label) for label in self._kmeans.labels_]

    def cluster_summary(self) -> dict[int, dict]:
        clusters: dict[int, list[EvaluationResult]] = defaultdict(list)
        for i, label in enumerate(self._labels):
            clusters[label].append(self._results[i])
        summaries: dict[int, dict] = {}
        for label, cluster_results in sorted(clusters.items()):
            failure_types = Counter(r.failure_type.value for r in cluster_results)
            summaries[label] = {
                "size": len(cluster_results),
                "avg_score": sum(r.score for r in cluster_results) / len(cluster_results),
                "dominant_failure_type": failure_types.most_common(1)[0][0],
                "failure_type_distribution": dict(failure_types),
            }
        return summaries
