"""Classification metrics (s04 version)."""
from __future__ import annotations

import numpy as np
from typing import Union

from ..utils.validation import check_consistent_length


def accuracy_score(y_true, y_pred) -> float:
    """Accuracy = mean(y_true == y_pred)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    """Build a (k, k) confusion matrix where k = len(labels).

    The union of ``y_true``, ``y_pred`` and ``labels`` is tokenized so that
    classes missing from the sample still get a zero row/column.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(labels)
    check_consistent_length(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred, labels]))
    class_to_index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_index[t], class_to_index[p]] += 1
    return cm
