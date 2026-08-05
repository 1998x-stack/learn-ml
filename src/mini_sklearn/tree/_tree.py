"""Recursive-partitioning decision trees (s12 / v0.3.4-5).

The core data structure is a binary ``TreeNode``.  ``fit`` grows one by
recursion:

    ``grow(parent) -> best_split -> grow(left child) + grow(right child)``

At every node we try, for each feature, every mid-point between two consecutive
*distinct* sorted values, and keep the single ``(feature, threshold)`` whose
split minimises the child-weighted impurity

    split_impurity = (n_L / n) * impurity(L) + (n_R / n) * impurity(R).

``impurity`` is gini or entropy for the classifier, and variance (=MSE) for the
regressor.  Leaves store a ``value`` (majority class / weighted mean target).
Recursion is guarded by ``max_depth`` and ``min_samples_split``, keeping the
tree shallow on the chapter's small synthetic data so it never blows Python's
``sys.setrecursionlimit``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]


# ---------------------------------------------------------------------------
# Tree primitive
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    """One node of a learned decision tree.

    Internal nodes carry ``feature_index`` / ``threshold`` and two children;
    leaves carry a prediction ``value``.  Every node records ``n_samples``
    (weighted count) and its own ``impurity`` so ``feature_importances_`` can
    be derived after fitting.
    """

    value: Optional[object] = None
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    n_samples: float = 0.0
    impurity: float = 0.0


# ---------------------------------------------------------------------------
# Classification impurities
# ---------------------------------------------------------------------------
def _gini(counts: np.ndarray) -> float:
    """``1 - sum_k p_k^2`` over weighted class proportions."""
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    p = counts / total
    return float(1.0 - (p * p).sum())


def _entropy(counts: np.ndarray) -> float:
    """``-sum_k p_k log2(p_k)``; 0 for a pure node."""
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    p = counts / total
    p = p[p > 0.0]
    return float(-(p * np.log2(p)).sum())


def _class_impurity(counts: np.ndarray, criterion: str) -> float:
    return _gini(counts) if criterion == "gini" else _entropy(counts)


def _weighted_counts(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-class weighted counts for an integer-coded label array."""
    n_classes = int(y.max()) + 1
    return np.bincount(y.astype(int), weights=w, minlength=n_classes)


# ---------------------------------------------------------------------------
# Regression impurity
# ---------------------------------------------------------------------------
def _variance(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted variance (per-node MSE impurity) of a leaf's targets."""
    total = w.sum()
    if total <= 0.0:
        return 0.0
    mean = (w * y).sum() / total
    return float((w * (y - mean) ** 2).sum() / total)


# ---------------------------------------------------------------------------
# Shared recursive-tree machinery
# ---------------------------------------------------------------------------
class _BaseTree(BaseEstimator):
    """Shared splitting / growing / importance logic.

    Subclasses implement two primitives:

    * ``_impurity(y, w)`` — impurity of one label/target array,
    * ``_node_from(y, w)`` — build a leaf ``TreeNode`` carrying ``value``.

    Everything else (split search, recursion, ``predict``, ``fit`` bookkeeping)
    lives here.
    """

    max_depth: Optional[int] = None
    min_samples_split: int = 2

    def _impurity(self, y: np.ndarray, w: np.ndarray) -> float:
        raise NotImplementedError

    def _node_from(self, y: np.ndarray, w: np.ndarray) -> TreeNode:
        raise NotImplementedError

    def _fit(self, X, y, w) -> "_BaseTree":
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._grow(X, y, w, depth=0)
        self.feature_importances_ = self._compute_importances(self.tree_)
        return self

    def _best_split(self, X, y, w) -> Tuple[Optional[int], Optional[float], float]:
        """Return ``(feature_index, threshold, best_impurity)`` for one node.

        For each feature the values are sorted; every mid-point between two
        consecutive *distinct* values is a candidate split.  The (feature,
        threshold) whose child-weighted impurity is smallest wins.
        """
        n_samples, n_features = X.shape
        parent = self._impurity(y, w)
        best = (None, None, parent)
        best_score = parent

        for f in range(n_features):
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]
            ys = y[order]
            ws = w[order]
            for i in range(1, n_samples):
                if xs[i] <= xs[i - 1]:
                    continue  # equal values -> no valid threshold between them
                threshold = (xs[i - 1] + xs[i]) / 2.0
                imp_left = self._impurity(ys[:i], ws[:i])
                imp_right = self._impurity(ys[i:], ws[i:])
                imp = (i / n_samples) * imp_left + (
                    (n_samples - i) / n_samples
                ) * imp_right
                if imp < best_score:
                    best_score = imp
                    best = (f, threshold, imp)
        return best

    def _grow(self, X, y, w, depth: int) -> TreeNode:
        n_samples = X.shape[0]
        node = self._node_from(y, w)
        node.n_samples = float(w.sum())

        # stop growing: pure-enough, at depth limit, or too few samples
        if node.impurity == 0.0:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n_samples < self.min_samples_split:
            return node
        if len(np.unique(y)) < 2:
            return node

        feature_idx, threshold, _ = self._best_split(X, y, w)
        if feature_idx is None:
            return node  # no split improves impurity

        left_mask = X[:, feature_idx] <= threshold
        if left_mask.all() or not left_mask.any():
            return node  # degenerate split that would not split anything

        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._grow(X[left_mask], y[left_mask], w[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], w[~left_mask], depth + 1)
        return node

    def _compute_importances(self, root: TreeNode) -> np.ndarray:
        """Total normalised weighted impurity decrease per feature.

        At every internal node the impurity reduction contributed by its split
        feature is ``(n_node / n_root) * (node.impurity - split_impurity)``;
        these accumulate per feature and are normalised so the vector sums to 1.
        """
        gains = np.zeros(self.n_features_in_)
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left is None or node.right is None:
                continue
            split_impurity = (
                node.left.n_samples / node.n_samples * node.left.impurity
                + node.right.n_samples / node.n_samples * node.right.impurity
            )
            gain = (node.n_samples / root.n_samples) * (
                node.impurity - split_impurity
            )
            gains[node.feature_index] += gain
            stack.append(node.left)
            stack.append(node.right)
        total = gains.sum()
        if total > 0.0:
            gains = gains / total
        return gains

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> object:
        while node.left is not None and node.right is not None:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


# ---------------------------------------------------------------------------
# Classification tree (s12 / v0.3.4 gini, v0.3.5 +entropy)
# ---------------------------------------------------------------------------
class DecisionTreeClassifier(_BaseTree, ClassifierMixin):
    """Recursively-partitioned classifier minimising gini / entropy impurity.

    v0.3.4 ships ``criterion="gini"`` with ``gini = 1 - sum p^2``;
    v0.3.5 adds ``criterion="entropy"`` ("-sum p log2 p").  Each leaf stores
    the majority class; ``predict`` walks root->leaf.  A split picks the
    ``(feature, threshold)`` that minimises the child-weighted gini/entropy.

    ``sample_weight`` is accepted so downstream AdaBoost (Task 4) can re-fit
    this tree on per-sample weights: gini counts, split gains and the leaf
    majority are then computed on *weighted* class frequencies.  Splitting is
    fully deterministic — ties are broken by the lowest feature index / lowest
    threshold first, so ``random_state`` is accepted for API compatibility (and
    future tie-breakers) but not needed to reproduce a run.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def _impurity(self, y: np.ndarray, w: np.ndarray) -> float:
        counts = _weighted_counts(y, w)
        return _class_impurity(counts, self.criterion)

    def _node_from(self, y: np.ndarray, w: np.ndarray) -> TreeNode:
        counts = _weighted_counts(y, w)
        value = int(np.argmax(counts))
        return TreeNode(
            value=value,
            n_samples=w.sum(),
            impurity=_class_impurity(counts, self.criterion),
        )

    def fit(self, X, y, sample_weight=None) -> "DecisionTreeClassifier":
        X, y = check_X_y(X, y)
        w = _resolve_sample_weight(sample_weight, X.shape[0])
        self.classes_ = np.unique(y)
        self._fit(X, y, w)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["tree_", "classes_", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            out[i] = self._predict_row(row, self.tree_)
        return out


# ---------------------------------------------------------------------------
# Regression tree (v0.3.5)
# ---------------------------------------------------------------------------
class DecisionTreeRegressor(_BaseTree, RegressorMixin):
    """Recursively-partitioned regressor minimising within-leaf variance (MSE).

    Each leaf stores the weighted mean of its targets; the node impurity is the
    within-leaf variance, and a split picks the ``(feature, threshold)`` with
    the lowest child-weighted MSE — the regression analogue of the
    classifier's impurity split.
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def _impurity(self, y: np.ndarray, w: np.ndarray) -> float:
        return _variance(y, w)

    def _node_from(self, y: np.ndarray, w: np.ndarray) -> TreeNode:
        total = w.sum()
        value = float((w * y).sum() / total) if total > 0.0 else 0.0
        return TreeNode(value=value, n_samples=total, impurity=_variance(y, w))

    def fit(self, X, y, sample_weight=None) -> "DecisionTreeRegressor":
        X, y = check_X_y(X, y)
        w = _resolve_sample_weight(sample_weight, X.shape[0])
        self._fit(X, y, w)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["tree_", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        out = np.empty(X.shape[0], dtype=float)
        for i, row in enumerate(X):
            out[i] = self._predict_row(row, self.tree_)
        return out


# ---------------------------------------------------------------------------
# shared helper
# ---------------------------------------------------------------------------
def _resolve_sample_weight(sample_weight, n_samples: int) -> np.ndarray:
    """Return an ``(n_samples,)`` float weight vector (default all-ones)."""
    if sample_weight is None:
        return np.ones(n_samples)
    w = np.asarray(sample_weight, dtype=float)
    if w.shape[0] != n_samples:
        raise ValueError("sample_weight and X have incompatible lengths.")
    return w
