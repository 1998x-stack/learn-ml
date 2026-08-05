"""mini_sklearn.metrics — regression and classification metrics (s04 / v0.1.5)."""

from ._regression import mean_absolute_error, mean_squared_error, r2_score
from ._classification import accuracy_score, confusion_matrix

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "confusion_matrix",
]
