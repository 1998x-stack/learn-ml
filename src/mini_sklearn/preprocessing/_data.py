"""Preprocessing: StandardScaler / MinMaxScaler / SimpleImputer (s05 / v0.2.x)."""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_array, check_is_fitted


class StandardScaler(BaseEstimator, TransformerMixin):
    """Center & scale each feature to zero mean / unit variance.

    fit stores train statistics; transform REUSES them (never recomputes on the
    transformed array). Stores mean_, scale_. y is accepted for the Transformer
    protocol but ignored.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        Xt = X - self.mean_ if self.with_mean else X
        return Xt / self.scale_ if self.with_std else Xt

    def inverse_transform(self, Xt) -> np.ndarray:
        check_is_fitted(self, ["mean_", "scale_"])
        Xt = np.asarray(Xt, dtype=float)
        X = Xt * self.scale_ if self.with_std else Xt
        return X + self.mean_ if self.with_mean else X


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Scale each feature to a given range, by default [0, 1].

    fit stores per-feature min_ & scale_ (= range); transform REUSES them.
    A feature with zero range gets scale_ = 1.0 (no scaling). Stores min_,
    scale_, n_features_in_.
    """

    def __init__(self, with_min: float = 0.0, with_max: float = 1.0):
        self.with_min = with_min
        self.with_max = with_max

    def fit(self, X, y=None):
        X = check_array(X)
        self.min_ = X.min(axis=0)
        scale = X.max(axis=0) - self.min_
        # guard zero-range features: leave them untouched (factor of 1.0)
        scale = np.where(scale == 0, 1.0, scale)
        self.scale_ = scale
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["min_", "scale_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        span = self.with_max - self.with_min
        return (X - self.min_) / self.scale_ * span + self.with_min

    def inverse_transform(self, Xt) -> np.ndarray:
        check_is_fitted(self, ["min_", "scale_"])
        Xt = np.asarray(Xt, dtype=float)
        span = self.with_max - self.with_min
        return (Xt - self.with_min) / span * self.scale_ + self.min_


class SimpleImputer(BaseEstimator, TransformerMixin):
    """Fill missing (NaN) values in each column with a univariate statistic.

    STRATEGIES: "mean" (column mean, default) or "median". Statistics are
    computed on the non-missing entries during fit and reused by transform.
    Stores statistics_ (row vector of per-column fill values), n_features_in_.
    """

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    def fit(self, X, y=None):
        X = check_array(X, allow_nan=True)
        self.n_features_in_ = X.shape[1]
        if self.strategy == "mean":
            statistics = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            statistics = np.nanmedian(X, axis=0)
        else:
            raise ValueError(f"Invalid strategy {self.strategy!r}; expected 'mean' or 'median'")
        self.statistics_ = statistics
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["statistics_"])
        X = check_array(X, allow_nan=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        missing = np.isnan(X)
        X = X.copy()
        X[missing] = np.take(self.statistics_, np.where(missing)[1])
        return X
