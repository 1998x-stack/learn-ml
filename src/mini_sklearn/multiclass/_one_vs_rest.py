"""One-vs-Rest multiclass meta-estimator (s09 / v0.2.8).

A **meta-estimator** wraps a base estimator instead of being a new learning
algorithm itself.  ``OneVsRestClassifier`` solves a k-class problem by fitting
``k`` independent binary classifiers — one per class, each deciding
"is this class `c`?" vs. "anything else" — and then choosing for every sample
the class whose binary model is most confident.

The word *clone* matters: each binary model must be a fresh, independent
copy sharing the base's hyperparameters but with its own untouched weights.
Re-using a single fitted model across classes would couple their training, so
we clone via ``from ..base import clone`` for k independent fits.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, clone
from ..utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["OneVsRestClassifier"]


class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-rest strategy: fit a binary clone of ``estimator`` per class.

    ``fit`` fits ``len(classes_)`` clones, the i-th on ``(X, y == classes_[i])``.
    ``predict`` gathers each clone's positive-class probability
    ``predict_proba(X)[:, 1]`` into an ``(n, k)`` score matrix and returns
    ``classes_[argmax]`` per row (ties fall to the lower class id).
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep: bool = True) -> dict:
        """Return ``{"estimator": <the base instance>}``.

        The wrapped estimator's ``estimator__*`` nested params are intentionally
        **not** surfaced here: the framework ``clone`` (s02) reconstructs an
        estimator via ``__class__(**get_params())``, so nested keys would break
        it. Teaching keeps the meta-estimator's surface minimal and correct.
        """
        return {"estimator": self.estimator}

    def set_params(self, **params):
        """Set the wrapped estimator param, or replace ``estimator`` itself."""
        if "estimator" in params:
            self.estimator = params["estimator"]
        for key in params:
            if key != "estimator" and hasattr(self, key):
                setattr(self, key, params[key])
            elif key != "estimator":
                raise ValueError(f"Invalid parameter {key!r}")
        return self

    def fit(self, X, y) -> "OneVsRestClassifier":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError(
                f"OneVsRestClassifier needs at least 2 classes, got {self.classes_.size}."
            )
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        for c in self.classes_:
            y_bin = (y == c).astype(float)
            est = clone(self.estimator)
            est.fit(X, y_bin)
            self.estimators_.append(est)
        return self

    def _score_matrix(self, X) -> np.ndarray:
        """Stack each clone's positive-class probability into an (n, k) matrix."""
        check_array(X)
        if hasattr(X, "shape") and X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        scores = np.empty((X.shape[0], len(self.estimators_)), dtype=float)
        for j, est in enumerate(self.estimators_):
            proba = est.predict_proba(X)
            # column 1 = positive-class probability ("is this class c vs rest")
            scores[:, j] = proba[:, 1]
        return scores

    def predict_proba(self, X) -> np.ndarray:
        """Return per-class scores normalised so rows sum to 1 (shape (n, k))."""
        scores = self._score_matrix(X)
        row_sum = scores.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        return scores / row_sum

    def predict(self, X) -> np.ndarray:
        """Return the class label with the highest positive score per row."""
        check_is_fitted(self, ["estimators_", "classes_", "n_features_in_"])
        X = check_array(X)
        scores = self._score_matrix(X)
        # argmax picks the highest score per row; np.argmax returns the first
        # (lowest index) among ties, which matches the lower-class-id tie rule.
        best = np.argmax(scores, axis=1)
        return self.classes_[best]
