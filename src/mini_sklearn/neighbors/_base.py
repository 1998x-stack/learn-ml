"""K-nearest-neighbors estimators: lazy learning (s10 / v0.3.2).

The defining property of a *lazy* learner: ``fit`` does no learning — it only
**stores** the training data.  All of the work happens in ``predict``, which
searches the stored points for the ``k`` nearest neighbours of each probe and
aggregates their labels (majority class for a classifier, mean for a
regressor).
"""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..metrics import euclidean_distances, manhattan_distances
from ..utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor"]

# ``cosine`` is intentionally NOT supported: it is a similarity (larger = more
# alike), so argsorting ascending would treat the *least* similar as "nearest".
_METRICS = {
    "euclidean": euclidean_distances,
    "manhattan": manhattan_distances,
}


def _neighbor_indices(X, X_fit, k: int, metric: str) -> np.ndarray:
    """Return indices of each row of ``X`` into ``X_fit`` of its k nearest.

    ``k`` nearest means the ``k`` smallest pair-wise distances.  For the small
    teaching datasets here we argsort the full ``(n_x, n_fit)`` matrix and take
    the first ``k`` columns per row (ties resolve to the *first* index
    encountered, which keeps the result deterministic).
    """
    if metric not in _METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}; choose from {sorted(_METRICS)}."
        )
    distance_fn = _METRICS[metric]
    D = distance_fn(X, X_fit)
    return np.argsort(D, axis=1)[:, :k]


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """k-nearest-neighbour classifier with majority-vote prediction."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X, y) -> "KNeighborsClassifier":
        X, y = check_X_y(X, y)
        self._fit_X = X
        self._fit_y = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["_fit_X", "_fit_y", "classes_", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        inds = _neighbor_indices(X, self._fit_X, self.n_neighbors, self.metric)
        nbr_y = self._fit_y[inds]  # (n_x, k)
        if self.weights == "distance":
            w = _inverse_distance_weights(X, self._fit_X, inds, self.metric)
        else:
            w = np.ones_like(nbr_y, dtype=float)
        # weighted vote: argmax over classes of the summed weights per class
        pred = []
        for row, row_w in zip(nbr_y, w):
            votes = np.zeros(self.classes_.size, dtype=float)
            for c_ix, c in enumerate(self.classes_):
                votes[c_ix] = row_w[row == c].sum()
            pred.append(self.classes_[np.argmax(votes)])
        return np.asarray(pred, dtype=self.classes_.dtype)


class KNeighborsRegressor(BaseEstimator, RegressorMixin):
    """k-nearest-neighbour regressor: predict the mean of the k neighbours."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X, y) -> "KNeighborsRegressor":
        X, y = check_X_y(X, y)
        self._fit_X = X
        self._fit_y = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["_fit_X", "_fit_y", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        inds = _neighbor_indices(X, self._fit_X, self.n_neighbors, self.metric)
        nbr_y = self._fit_y[inds].astype(float)
        if self.weights == "distance":
            w = _inverse_distance_weights(X, self._fit_X, inds, self.metric)
        else:
            w = np.ones_like(nbr_y, dtype=float)
        return (nbr_y * w).sum(axis=1) / w.sum(axis=1)


def _inverse_distance_weights(X, X_fit, inds, metric: str) -> np.ndarray:
    """Weights ``1/d`` per neighbour; an exact match (d=0) caps the weight.

    Neighbours closer to the query should vote louder: weight is the inverse
    of distance.  A ``d == 0`` duplicate would yield ``inf`` and dominate, so
    we clamp a zero distance to a tiny floor before inverting.
    """
    distance_fn = _METRICS[metric]
    D = distance_fn(X, X_fit)
    d = np.take_along_axis(D, inds, axis=1)
    d_floor = np.where(d > 0.0, d, 1e-12)
    return 1.0 / d_floor
