"""mini_sklearn.neighbors — lazy-learning nearest-neighbour estimators (s10 / v0.3.2).

K-nearest neighbours is the framework's first **lazy** model: ``fit`` merely
stores the training data, and ``predict`` does the real work by searching the
stored points for the ``k`` nearest neighbours of each query.
"""

from ._base import KNeighborsClassifier, KNeighborsRegressor

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor"]
