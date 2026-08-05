"""Estimator core: BaseEstimator + Mixins + clone(). (s02 / v0.1.3)"""
from __future__ import annotations

class BaseEstimator:
    """Introspect __init__ signature → get_params / set_params.

    Enforces the invariant: __init__ may only assign hyperparameters
    (no learned attributes, which must carry a trailing underscore and
    only appear after fit()).
    """

    def get_params(self, deep: bool = True) -> dict:
        from inspect import signature
        out: dict = {}
        for name in signature(self.__init__).parameters:
            if name == "self":
                continue
            out[name] = getattr(self, name)
        return out

    def set_params(self, **params) -> "BaseEstimator":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key!r}")
            setattr(self, key, value)
        return self


class ClassifierMixin:
    _estimator_type = "classifier"

class RegressorMixin:
    _estimator_type = "regressor"

class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        return self.fit(X, y, **fit_params).transform(X)


def clone(estimator: BaseEstimator) -> BaseEstimator:
    """Return a fresh untrained estimator with identical hyperparameters."""
    return estimator.__class__(**estimator.get_params())
