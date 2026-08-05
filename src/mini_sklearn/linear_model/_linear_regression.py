"""Linear regression estimators: LinearRegression, Ridge (s06 / v0.2.3)."""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, RegressorMixin
from ..utils.validation import check_array, check_is_fitted
from ._base import _add_intercept, _check_input


class LinearRegression(BaseEstimator, RegressorMixin):
    """Closed-form least-squares linear regression via the pseudoinverse.

    Solves Xw = y using np.linalg.pinv (least-squares pseudoinverse). When
    fit_intercept=True a column of ones is prepended so the bias is solved as
    part of the augmented weight vector. coef_ holds the feature weights
    (shape (n_features,)); intercept_ is the scalar bias term.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> "LinearRegression":
        X, y = _check_input(X, y)
        self.n_features_in_ = X.shape[1]
        if self.fit_intercept:
            Xa = _add_intercept(X)
            w = np.linalg.pinv(Xa) @ y
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            w = np.linalg.pinv(X) @ y
            self.coef_ = w
            self.intercept_ = 0.0
        self.coef_ = np.asarray(self.coef_, dtype=float)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return X @ self.coef_ + self.intercept_


class Ridge(BaseEstimator, RegressorMixin):
    """Normal-equation linear regression with L2 (ridge) regularization.

    Closed form w = (X^T X + alpha I)^{-1} X^T y. With fit_intercept=True the
    augmented system (bias column included) is solved, so the intercept is
    regularized too, mirroring sklearn's solver on centered data.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> "Ridge":
        X, y = _check_input(X, y)
        self.n_features_in_ = X.shape[1]
        if self.fit_intercept:
            Xa = _add_intercept(X)
        else:
            Xa = X
        XtX = Xa.T @ Xa
        # add alpha along the diagonal (regularizes every column, incl. bias)
        XtX = XtX + self.alpha * np.eye(Xa.shape[1])
        w = np.linalg.solve(XtX, Xa.T @ y)
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = np.asarray(w[1:], dtype=float)
        else:
            self.coef_ = np.asarray(w, dtype=float)
            self.intercept_ = 0.0
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return X @ self.coef_ + self.intercept_
