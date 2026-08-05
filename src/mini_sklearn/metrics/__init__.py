"""mini_sklearn.metrics — regression, classification & pairwise metrics (s04 / s10)."""

from ._classification import accuracy_score, confusion_matrix
from ._pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from ._regression import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "confusion_matrix",
    "euclidean_distances",
    "manhattan_distances",
    "cosine_similarity",
]

