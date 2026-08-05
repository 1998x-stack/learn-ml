"""mini_sklearn.multiclass — multiclass strategies & label encoding (s09 / v0.2.8).

``OneVsRestClassifier`` is the framework's first **meta-estimator**: it wraps
any binary classifier and turns it into a multiclass classifier without
changing the base model at all.
"""

from ._encoders import LabelEncoder
from ._one_vs_rest import OneVsRestClassifier

__all__ = ["LabelEncoder", "OneVsRestClassifier"]
