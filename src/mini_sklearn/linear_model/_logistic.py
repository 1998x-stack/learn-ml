"""Probabilistic linear classifier: LogisticRegression via GD (s08 / v0.2.6-7).

Unlike the closed-form least-squares of s06, logistic regression has no closed
form. It optimizes the **binary cross-entropy** by full-batch gradient descent:
a linear score z = X @ coef + b is mapped through a sigmoid to a probability
p = 1/(1+exp(-z)), and the weights are updated by w -= lr * (1/n) Xa.T @ (p - y).
Binary (2-class) only; multiclass arrives in s09.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..utils.validation import check_array, check_is_fitted, check_random_state
from ._base import _add_intercept, _check_input


def _sigmoid(z) -> np.ndarray:
    """Numerically stable sigmoid 1 / (1 + exp(-z)).

    Branch on the sign of z so exp(-z) never overflows (z very negative) and the
    naive reciprocal never returns inf. For z >= 0 -> 1/(1+exp(-z)); else use
    exp(z)/(1+exp(z)).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def _binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy -mean(y log p + (1-y) log(1-p)).

    p is clipped out of {0, 1} so log() never receives 0 (which would give
    -inf and destroy the finite mean).
    """
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary logistic regression fit by full-batch gradient descent.

    Minimizes binary cross-entropy over the linear scores; the gradient wrt the
    score is exactly (p - y), giving the GD update
        w -= learning_rate * (1/n) * Xa.T @ (p - y).
    Stores coef_ (shape (n_features,)), scalar intercept_, classes_ (sorted
    unique labels) plus loss_history_ / n_iter_.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        random_state: int = 0,
        fit_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def _loss(self, y: np.ndarray, p: np.ndarray) -> float:
        return _binary_cross_entropy(y, p)

    def fit(self, X, y) -> "LogisticRegression":
        X, y = _check_input(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                f"LogisticRegression supports binary classification only; "
                f"got {self.classes_.size} classes."
            )
        rng = check_random_state(self.random_state)
        # re-code labels {a, b} -> {0, 1}; class 1 = the max class
        y_bin = (y == self.classes_[1]).astype(float)
        Xa = _add_intercept(X) if self.fit_intercept else X
        n = Xa.shape[0]
        w = rng.standard_normal(Xa.shape[1]) * 1e-3
        loss_history = [self._loss(y_bin, _sigmoid(Xa @ w))]
        for _ in range(self.max_iter):
            p = _sigmoid(Xa @ w)
            grad = (1.0 / n) * (Xa.T @ (p - y_bin))
            w = w - self.learning_rate * grad
            loss_history.append(self._loss(y_bin, _sigmoid(Xa @ w)))
        self.loss_history_ = np.asarray(loss_history, dtype=float)
        self.n_iter_ = self.max_iter
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = np.asarray(w[1:], dtype=float)
        else:
            self.coef_ = np.asarray(w, dtype=float)
            self.intercept_ = 0.0
        return self

    def decision_function(self, X) -> np.ndarray:
        """Raw scores X @ coef_ + intercept_ (shape (n,))."""
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        """Return an (n, 2) array [1 - p, p] with p = P(class_1); rows sum to 1."""
        z = self.decision_function(X)
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        """Return the most probable class label per sample (ties -> class_1)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
