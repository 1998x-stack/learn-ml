"""mini_sklearn.tree — recursive-partitioning decision trees (s12 / v0.3.4-5).

Decision trees are the framework's first **recursive** model: instead of a
single global rule, ``fit`` grows a binary tree where each internal node splits
the data on one feature threshold, chosen to minimise a child-weighted
impurity (gini or entropy for classification, MSE for regression).  A single
split is weak; recursively splitting builds arbitrarily complex decision
boundaries — the s12 slogan ``一次划分很弱，递归划分形成复杂边界``.
"""

from ._tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]
