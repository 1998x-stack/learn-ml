"""Iterative optimization estimators: GDRegressor, SGDRegressor (s07 / v0.2.4-5)."""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, RegressorMixin
from ..utils.validation import check_array, check_is_fitted, check_random_state
from ._base import _add_intercept, _check_input


class GDRegressor(BaseEstimator, RegressorMixin):
    """Full-batch gradient descent for linear regression.

    Instead of the closed-form pseudoinverse (s06), fit() minimizes the MSE
    by iteratively stepping w -= lr * grad over the *full* dataset each
    iteration. grads = (2 / n) * Xa.T @ (Xa @ w - y). With fit_intercept=True
    a column of ones is prepended so the bias is learned as part of w_aug
    = [intercept, coef...], then split back into coef_/intercept_ like s06.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _loss(self, Xa: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        resid = Xa @ w - y
        return float(np.mean(resid ** 2))

    def fit(self, X, y) -> "GDRegressor":
        X, y = _check_input(X, y)
        self.n_features_in_ = X.shape[1]
        Xa = _add_intercept(X) if self.fit_intercept else X
        n, d = Xa.shape
        w = np.zeros(d)
        loss = self._loss(Xa, w, y)
        loss_history = [loss]
        n_iter = 0
        converged = False
        for _ in range(self.max_iter):
            grad = (2.0 / n) * (Xa.T @ (Xa @ w - y))
            w = w - self.learning_rate * grad
            loss = self._loss(Xa, w, y)
            loss_history.append(loss)
            n_iter += 1
            if np.linalg.norm(grad) <= self.tol:
                converged = True
                break
        self.loss_history_ = np.asarray(loss_history, dtype=float)
        self.n_iter_ = n_iter
        self.converged_ = converged
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


class SGDRegressor(BaseEstimator, RegressorMixin):
    """Stochastic gradient descent regression (single-sample steps per epoch).

    Each epoch shuffles the samples (seeded via check_random_state) then updates
    weights on each single-sample gradient grad_i = 2 * xa_i * (xa_i @ w - y_i).
    Noisy but scales to datasets far too big for a full-batch step.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 0,
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def _mse(self, Xa: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        resid = Xa @ w - y
        return float(np.mean(resid ** 2))

    def fit(self, X, y) -> "SGDRegressor":
        X, y = _check_input(X, y)
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        Xa = _add_intercept(X) if self.fit_intercept else X
        n = Xa.shape[0]
        w = np.zeros(Xa.shape[1])
        loss_history = [self._mse(Xa, w, y)]
        for _ in range(self.max_iter):
            order = rng.permutation(n)
            for i in order:
                x_i = Xa[i]
                grad_i = 2.0 * x_i * (x_i @ w - y[i])
                w = w - self.learning_rate * grad_i
            loss_history.append(self._mse(Xa, w, y))
        self.loss_history_ = np.asarray(loss_history, dtype=float)
        self.n_iter_ = self.max_iter
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
