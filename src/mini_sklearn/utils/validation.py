"""Input and state validation utilities (s03 version)."""
from __future__ import annotations
import numbers
from typing import Union
import numpy as np


def check_array(X, dtype=float, ensure_2d: bool = True, allow_nan: bool = False) -> np.ndarray:
    X_a = np.asarray(X, dtype=dtype)
    if ensure_2d and X_a.ndim != 2:
        raise ValueError(f"X must be 2d, got ndim={X_a.ndim}")
    if not allow_nan and (np.isnan(X_a).any() or np.isinf(X_a).any()):
        raise ValueError("X contains NaN or infinity; they are not allowed by default")
    return X_a


def check_consistent_length(*arrays) -> bool:
    lengths = {len(np.asarray(a)) for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"Found input variables with inconsistent lengths (length mismatch): {lengths}")
    return True


def check_X_y(X, y) -> tuple:
    X_a = check_array(X)
    y_a = np.asarray(y)
    check_consistent_length(X_a, y_a)
    return X_a, y_a


def check_is_fitted(estimator, attributes: Union[str, list], msg: str | None = None) -> None:
    if isinstance(attributes, str):
        attributes = [attributes]
    missing = [a for a in attributes if not hasattr(estimator, a)]
    if missing:
        raise ValueError(msg or f"This {type(estimator).__name__} instance is not fitted yet: missing {missing}")


def check_random_state(seed) -> np.random.Generator:
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a numpy.random.Generator")