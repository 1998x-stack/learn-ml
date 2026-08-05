"""Label encoding for classification targets (s09 / v0.2.8).

Multiclass algorithms (one-vs-rest in s09) need integer class ids
``0..k-1`` to index per-class models, but real datasets carry arbitrary
labels such as strings or uneven integers. ``LabelEncoder`` centralises the
mapping: it learns a sorted set of classes on ``fit`` and converts both ways
between original labels and the dense ``0..k-1`` codes.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, TransformerMixin

__all__ = ["LabelEncoder"]


def _as_1d(y) -> np.ndarray:
    """Return y as a flat 1d ndarray, rejecting multi-column input."""
    arr = np.asarray(y)
    if arr.ndim > 1:
        raise ValueError(f"y must be 1d, got ndim={arr.ndim}")
    return arr.ravel()


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Map arbitrary labels to dense integers ``0..k-1`` (and back).

    ``fit(y)`` learns ``classes_`` = the sorted unique labels.  ``transform``
    replaces each label with its index in that sorted order; ``inverse_transform``
    recovers the original labels from integer codes ``0..k-1``.
    """

    def get_params(self, deep: bool = True) -> dict:
        # LabelEncoder carries no hyperparameters; kept so it remains an Estimator.
        return {}

    def fit(self, y) -> "LabelEncoder":
        y_flat = _as_1d(y)
        self.classes_ = np.unique(y_flat)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y) -> np.ndarray:
        """Return the encoded integer ``0..k-1`` for each label in ``y``."""
        if not hasattr(self, "classes_"):
            raise ValueError(
                "This LabelEncoder instance is not fitted yet; call 'fit' first."
            )
        y_flat = _as_1d(y)
        try:
            return np.array([self._class_to_idx[v] for v in y_flat], dtype=int)
        except KeyError:
            unseen = sorted({v for v in y_flat if v not in self._class_to_idx})
            raise ValueError(
                f"y contains labels not seen during fit: {unseen[:5]}"
            ) from None

    def inverse_transform(self, y) -> np.ndarray:
        """Return the original labels for integer codes ``y`` in 0..k-1."""
        if not hasattr(self, "classes_"):
            raise ValueError(
                "This LabelEncoder instance is not fitted yet; call 'fit' first."
            )
        y_int = np.asarray(y, dtype=int)
        if y_int.size and (y_int.min() < 0 or y_int.max() >= self.classes_.size):
            raise ValueError(
                "Encoded labels are out of the range [0, n_classes); were they "
                "produced by this instance?"
            )
        return self.classes_[y_int]
