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
    """Predicts failure probability for prompts using TF-IDF features."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self._classifier = LogisticRegression(max_iter=1000)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def train(self, results: list[EvaluationResult], threshold: int = 4) -> dict:
        texts = [r.prompt_text for r in results]
        labels = [1 if r.score >= threshold else 0 for r in results]
        X = self._vectorizer.fit_transform(texts)
        self._classifier.fit(X, labels)
        self._fitted = True
        import numpy as np
        min_class_count = int(np.bincount(labels).min()) if len(set(labels)) > 1 else 1
        cv_folds = min(5, len(results), max(2, min_class_count))
        scores = cross_val_score(self._classifier, X, labels, cv=cv_folds)
        return {
            "samples": len(results),
            "positive_rate": sum(labels) / max(1, len(labels)),
            "cv_accuracy": float(scores.mean()),
        }

    def predict(self, prompt_text: str) -> float:
        if not self._fitted:
            raise RuntimeError("Predictor not trained. Call train() first.")
        X = self._vectorizer.transform([prompt_text])
        return float(self._classifier.predict_proba(X)[0, 1])

    def predict_batch(self, texts: list[str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("Predictor not trained. Call train() first.")
        X = self._vectorizer.transform(texts)
        return [float(p) for p in self._classifier.predict_proba(X)[:, 1]]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self._vectorizer, "classifier": self._classifier}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._vectorizer = data["vectorizer"]
        self._classifier = data["classifier"]
        self._fitted = True
