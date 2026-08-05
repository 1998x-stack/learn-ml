"""Pairwise distance & similarity metrics (s10 / v0.3.1).

All functions return an ``(n_x, n_y)`` matrix of pair-wise values between
rows of ``X`` and rows of ``Y``.  ``Y=None`` means "vs. itself" (``Y=X``), so
the diagonal of a distance matrix is exactly ``0`` for any valid metric.

Vectorisation is the whole point of this chapter: a naive "Python double
for-loop over sample pairs" costs ``O(n_x * n_y * d)`` Python operations,
whereas NumPy broadcasting computes every pair in one expression
(``X[:, None, :] - Y[None, :, :]`` is ``(n_x, n_y, d)``).
"""
from __future__ import annotations

import numpy as np

from ..utils.validation import check_array

__all__ = [
    "euclidean_distances",
    "manhattan_distances",
    "cosine_similarity",
]


def _check_pairwise_inputs(X, Y) -> tuple[np.ndarray, np.ndarray]:
    """Validate both inputs; ``Y=None`` degenerates to ``X`` (self-pairing)."""
    X_a = check_array(X)
    if Y is None:
        Y_a = X_a
    else:
        Y_a = check_array(Y)
        if X_a.shape[1] != Y_a.shape[1]:
            raise ValueError(
                "Incompatible column counts: X has "
                f"{X_a.shape[1]} features, Y has {Y_a.shape[1]}."
            )
    return X_a, Y_a


def euclidean_distances(X, Y=None) -> np.ndarray:
    """Euclidean (L2) distance matrix between rows of X and rows of Y.

    ``np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))``
    Broadcasting turns the ``(n_x, d)`` and ``(n_y, d)`` inputs into a single
    ``(n_x, n_y, d)`` difference tensor, then reduces over the feature axis.
    """
    X_a, Y_a = _check_pairwise_inputs(X, Y)
    diff = X_a[:, None, :] - Y_a[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def manhattan_distances(X, Y=None) -> np.ndarray:
    """Manhattan (L1) distance matrix between rows of X and rows of Y."""
    X_a, Y_a = _check_pairwise_inputs(X, Y)
    diff = X_a[:, None, :] - Y_a[None, :, :]
    return np.abs(diff).sum(axis=-1)


def cosine_similarity(X, Y=None) -> np.ndarray:
    """Cosine similarity between rows of X and rows of Y.

    Normalises every row to unit norm, then matrix-dots.  For ``Y=X`` the
    diagonal is exactly ``1.0`` (a vector's cosine with itself).  Zero-norm
    rows are left untouched (returned similarity 0 for that row).
    """
    X_a, Y_a = _check_pairwise_inputs(X, Y)
    x_norm = np.linalg.norm(X_a, axis=1, keepdims=True)
    y_norm = np.linalg.norm(Y_a, axis=1, keepdims=True)
    x = np.divide(X_a, x_norm, out=np.zeros_like(X_a), where=x_norm > 0)
    y = np.divide(Y_a, y_norm, out=np.zeros_like(Y_a), where=y_norm > 0)
    return x @ y.T
