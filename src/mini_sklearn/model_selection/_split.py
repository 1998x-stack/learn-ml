"""Train/test data splitting (s04 version)."""
from __future__ import annotations

import numpy as np
from typing import List

from ..utils.validation import check_consistent_length, check_random_state


def train_test_split(
    *arrays, test_size: float = 0.25, random_state=None
) -> List[np.ndarray]:
    """Split arrays into train/test subsets along axis 0 with a shared permutation.

    Returns all train arrays first, then all test arrays, so that
    ``X_tr, y_tr, X_te, y_te = train_test_split(X, y, ...)`` unpacks correctly.
    """
    arrays = [np.asarray(a) for a in arrays]
    check_consistent_length(*arrays)
    if not arrays:
        raise ValueError("At least one array required as input to train_test_split")
    n = arrays[0].shape[0]
    rng = check_random_state(random_state)
    perm = rng.permutation(n)
    n_test = int(np.ceil(n * test_size))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return [a[train_idx] for a in arrays] + [a[test_idx] for a in arrays]
