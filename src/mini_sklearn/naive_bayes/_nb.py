"""Naive-Bayes classifiers: generative probability modelling (s11 / v0.3.3).

KNN (s10) judged a query by *distances* to stored neighbours.  Naive Bayes
flips to *probabilistic* reasoning: it models how each class **generates**
its features and predicts with the Bayes rule

    P(y | x)  ∝  P(y) × ∏_d P(x_d | y).

The word "naive" is the crux: it **assumes every feature is independent given
the class**, so the joint likelihood factorises into a product of per-feature
densities.  That strong assumption buys a simple, **fast and stable** learner
(`强假设可以换来简单、快速和稳定`).  Inference happens in **log space** —
`log_prior + Σ_d log P(x_d|y)` — so the product of many tiny probabilities
never underflows to zero.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["GaussianNB", "MultinomialNB", "BernoulliNB"]


def _stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-normalised softmax in log space -> probabilities in (0,1), rows sum 1.

    Subtract the row max before exponentiating so exp() never overflows; the
    result is invariant to the shift.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _predict_classes(joint_log_likelihood: np.ndarray, classes_: np.ndarray) -> np.ndarray:
    """Argmax over the (n_samples, n_classes) log-space table."""
    return classes_[np.argmax(joint_log_likelihood, axis=1)]


class GaussianNB(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes: continuous features as per-class Gaussians.

    ``fit`` records, for every class ``k`` and feature ``d``, the empirical mean
    ``theta_[d, k]`` and variance ``sigma_[d, k]`` (= Var + ``var_smoothing``, so
    a degenerate zero-variance feature never divides by zero).  Prediction sums
    the per-feature Gaussian log-pdfs in **log space**.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y) -> "GaussianNB":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.size

        class_count_ = np.empty(n_classes, dtype=float)
        theta_ = np.empty((n_features, n_classes), dtype=float)  # means (D, K)
        sigma_ = np.empty((n_features, n_classes), dtype=float)  # vars  (D, K)
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            class_count_[k] = Xc.shape[0]
            theta_[:, k] = Xc.mean(axis=0)
            sigma_[:, k] = Xc.var(axis=0) + self.var_smoothing

        self.epsilon_ = self.var_smoothing
        self.class_count_ = class_count_
        self.class_prior_ = class_count_ / n_samples
        self.class_log_prior_ = np.log(self.class_prior_)
        self.theta_ = theta_
        self.sigma_ = sigma_
        self.n_features_in_ = n_features
        return self

    def _check_is_fitted(self, X) -> np.ndarray:
        check_is_fitted(
            self, ["theta_", "sigma_", "class_log_prior_", "classes_", "n_features_in_"]
        )
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return X

    def joint_log_likelihood(self, X) -> np.ndarray:
        """Per-class ``log P(y) + sum_d log N(x_d; theta, sigma)`` (n, K)."""
        X = self._check_is_fitted(X)
        # Gaussian log-pdf per feature & class; broadcasting (n, D, K)
        log_pdf = (
            -0.5 * np.log(2.0 * np.pi * self.sigma_)[None, :, :]
            - 0.5 * (X[:, :, None] - self.theta_[None, :, :]) ** 2
            / self.sigma_[None, :, :]
        )
        return log_pdf.sum(axis=1) + self.class_log_prior_[None, :]

    def predict_proba(self, X) -> np.ndarray:
        return _stable_log_softmax(self.joint_log_likelihood(X))

    def predict(self, X) -> np.ndarray:
        return _predict_classes(
            self.joint_log_likelihood(X), self.classes_
        )


class MultinomialNB(BaseEstimator, ClassifierMixin):
    """Multinomial Naive Bayes: non-negative count features (e.g. word counts).

    The likelihood of feature ``d`` under class ``k`` is its empirical frequency
    smoothed by ``alpha`` (Laplace when ``alpha=1``).  The full ``predict_proba``
    runs in log space to keep counts stable.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y) -> "MultinomialNB":
        X, y = check_X_y(X, y)
        if (X < 0).any():
            raise ValueError("MultinomialNB expects non-negative feature counts.")
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.size

        feature_count_ = np.empty((n_classes, n_features), dtype=float)
        class_count_ = np.empty(n_classes, dtype=float)
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            class_count_[k] = Xc.shape[0]
            feature_count_[k] = Xc.sum(axis=0)

        class_log_prior_ = np.log(class_count_ / n_samples)
        # Laplace/pseudo-count smoothing, in log space
        smoothed = (feature_count_ + self.alpha) / (
            feature_count_.sum(axis=1, keepdims=True) + self.alpha * n_features
        )
        self.class_count_ = class_count_
        self.feature_count_ = feature_count_
        self.class_log_prior_ = class_log_prior_
        self.feature_log_prob_ = np.log(smoothed)
        self.n_features_in_ = n_features
        return self

    def _check_is_fitted(self, X) -> np.ndarray:
        check_is_fitted(
            self,
            ["feature_log_prob_", "class_log_prior_", "classes_", "n_features_in_"],
        )
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return X

    def joint_log_likelihood(self, X) -> np.ndarray:
        X = self._check_is_fitted(X)
        # feature_log_prob_ @ X.T is (K, n); transpose to (n, K)
        return (self.feature_log_prob_ @ X.T).T + self.class_log_prior_[None, :]

    def predict_proba(self, X) -> np.ndarray:
        return _stable_log_softmax(self.joint_log_likelihood(X))

    def predict(self, X) -> np.ndarray:
        return _predict_classes(
            self.joint_log_likelihood(X), self.classes_
        )


class BernoulliNB(BaseEstimator, ClassifierMixin):
    """Bernoulli Naive Bayes: binary / 0,1 indicator features (presence or absence)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y) -> "BernoulliNB":
        X, y = check_X_y(X, y)
        if not np.isin(X, [0.0, 1.0]).all():
            raise ValueError("BernoulliNB expects binary (0/1) features.")
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.size

        feature_count_ = np.empty((n_classes, n_features), dtype=float)
        class_count_ = np.empty(n_classes, dtype=float)
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            class_count_[k] = Xc.shape[0]
            feature_count_[k] = Xc.sum(axis=0)

        class_log_prior_ = np.log(class_count_ / n_samples)
        # smoothed P(feature present | class), alpha-signed for a two-outcome space
        feature_prob_ = (feature_count_ + self.alpha) / (
            class_count_[:, None] + 2.0 * self.alpha
        )
        neg_feature_prob_ = 1.0 - feature_prob_

        self.class_count_ = class_count_
        self.feature_count_ = feature_count_
        self.class_log_prior_ = class_log_prior_
        self.feature_log_prob_ = np.log(feature_prob_)
        self.feature_log_prob_neg_ = np.log(neg_feature_prob_)
        self.n_features_in_ = n_features
        return self

    def _check_is_fitted(self, X) -> np.ndarray:
        check_is_fitted(
            self,
            [
                "feature_log_prob_",
                "feature_log_prob_neg_",
                "class_log_prior_",
                "classes_",
                "n_features_in_",
            ],
        )
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return X

    def joint_log_likelihood(self, X) -> np.ndarray:
        X = self._check_is_fitted(X)
        pos = X @ self.feature_log_prob_.T
        neg = (1.0 - X) @ self.feature_log_prob_neg_.T
        return pos + neg + self.class_log_prior_[None, :]

    def predict_proba(self, X) -> np.ndarray:
        return _stable_log_softmax(self.joint_log_likelihood(X))

    def predict(self, X) -> np.ndarray:
        return _predict_classes(
            self.joint_log_likelihood(X), self.classes_
        )
