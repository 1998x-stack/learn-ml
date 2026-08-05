"""Synthetic data generators (s04 version)."""
from __future__ import annotations

import numpy as np
from typing import Tuple

from ..utils.validation import check_random_state


def make_regression(
    n_samples: int = 100,
    n_features: int = 10,
    noise: float = 0.0,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random linear regression problem: y = X @ coef + noise."""
    rng = check_random_state(random_state)
    X = rng.standard_normal((n_samples, n_features))
    coef = rng.normal(size=n_features)
    y = X @ coef + noise * rng.standard_normal(n_samples)
    return X, y


def make_classification(
    n_samples: int = 100,
    n_features: int = 10,
    n_classes: int = 2,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic classification problem: each class is a Gaussian blob."""
    rng = check_random_state(random_state)
    X_base = rng.standard_normal((n_samples, n_features))
    centers = rng.normal(size=(n_classes, n_features))
    classes = rng.integers(0, n_classes, size=n_samples)
    X = X_base + centers[classes]
    return X, classes
