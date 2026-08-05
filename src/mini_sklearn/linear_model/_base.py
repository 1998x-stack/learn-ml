"""Shared helpers for linear model estimators (s06 / v0.2.3)."""
from __future__ import annotations

import numpy as np

from ..utils.validation import check_X_y


def _check_input(X, y):
    """Validate (X, y) and return 2d float X and 1d float y.

    Reused by both LinearRegression and Ridge so the input contract lives in
    exactly one place.
    """
    X, y = check_X_y(X, y)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X, y


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X to absorb an intercept (bias) term.

    Solving X_aug @ w_aug = y then yields w_aug = [intercept, coef...].
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])
