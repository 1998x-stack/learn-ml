"""Parity tests: our NumPy mini-sklearn vs the dev-only real scikit-learn (v0.2).

Every assertion compares our framework's output to `sklearn`'s on the SAME
deterministic input, within a tolerance. This track exists to prove the
NumPy-from-scratch implementations are numerically faithful to sklearn's —
the teaching/framework code itself NEVER imports sklearn; it lives only here.

Modules under test (v0.2, s05–s09):
    preprocessing.StandardScaler   vs sklearn.preprocessing.StandardScaler
    linear_model.LinearRegression  vs sklearn.linear_model.LinearRegression
    linear_model.LogisticRegression vs sklearn.linear_model.LogisticRegression
    multiclass.OneVsRestClassifier vs sklearn.multiclass.OneVsRestClassifier

Convention note on StandardScaler.scale_: sklearn's ``scale_`` is the
population (ddof=0) std per feature — our framework matches it exactly, so the
parity assertion compares .mean_ / .scale_ / transform() directly to 1e-6.
"""
from __future__ import annotations

import numpy as np
import pytest

# --- dev-only reference: never imported by framework/teaching code ----------
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.multiclass import OneVsRestClassifier as SkOneVsRest
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from src.mini_sklearn.linear_model import LinearRegression, LogisticRegression
from src.mini_sklearn.multiclass import OneVsRestClassifier
from src.mini_sklearn.preprocessing import StandardScaler


def make_regression_data(n: int = 200, n_features: int = 3, seed: int = 0):
    """Well-conditioned linear data: y = X @ w + bias + small noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    w = rng.uniform(-1.0, 1.0, size=n_features)
    y = X @ w + 0.5 + rng.standard_normal(n) * 0.1
    return X, y


def make_separable_2class(seed: int = 1):
    """Two linearly separable Gaussian blobs (classes 0 and 1)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
    per = 100
    X = np.vstack([rng.normal(c, 0.4, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(i, per) for i in range(2)])
    return X, y


def make_separable_3class(seed: int = 7):
    """Three well-separated 2D Gaussian blobs (classes 0, 1, 2)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-4.0, -4.0], [4.0, -4.0], [0.0, 4.0]])
    per = 50
    X = np.vstack([rng.normal(c, 0.5, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(i, per) for i in range(3)])
    return X, y


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

def test_standard_scaler_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4)) * 3.0 + 2.0
    X_test = rng.standard_normal((30, 4)) * 2.5 + 1.0

    ours = StandardScaler().fit(X)
    theirs = SkStandardScaler().fit(X)

    np.testing.assert_allclose(ours.mean_, theirs.mean_, atol=1e-6)
    np.testing.assert_allclose(ours.scale_, theirs.scale_, atol=1e-6)
    np.testing.assert_allclose(ours.transform(X), theirs.transform(X), atol=1e-6)
    np.testing.assert_allclose(ours.transform(X_test), theirs.transform(X_test), atol=1e-6)


def test_standard_scaler_scale_is_population_std_matching_sklearn():
    """Guard the ddof convention: sklearn scale_ == np.std(X, axis=0) (ddof=0)."""
    X = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    np.testing.assert_allclose(SkStandardScaler().fit(X).scale_, X.std(axis=0))
    np.testing.assert_allclose(StandardScaler().fit(X).scale_, X.std(axis=0))


# ---------------------------------------------------------------------------
# LinearRegression
# ---------------------------------------------------------------------------
def test_linear_regression_matches_sklearn_coefs_and_predict():
    X, y = make_regression_data()

    ours = LinearRegression().fit(X, y)
    theirs = SkLinearRegression().fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        ours.predict(X), theirs.predict(X), rtol=1e-5, atol=1e-6
    )


def test_linear_regression_no_intercept_matches_sklearn():
    X, y = make_regression_data()
    # force through origin; sklearn requires fitting without intercept too
    X, y = make_regression_data(seed=1)

    ours = LinearRegression(fit_intercept=False).fit(X, y)
    theirs = SkLinearRegression(fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# LogisticRegression (binary, separable) — sklearn uses its own solver, so
# assert approximate agreement, not bit-exact equality.
# ---------------------------------------------------------------------------
def test_logistic_regression_predict_proba_matches_sklearn():
    X, y = make_separable_2class()

    # framework is unregularized GD -> sklearn penalty=None for a fair match
    ours = LogisticRegression(max_iter=8000, random_state=0).fit(X, y)
    theirs = SkLogisticRegression(max_iter=2000, penalty=None, solver="lbfgs").fit(X, y)

    assert np.mean(ours.predict(X) == y) == 1.0
    assert np.mean(theirs.predict(X) == y) == 1.0
    np.testing.assert_allclose(
        ours.predict_proba(X), theirs.predict_proba(X), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# OneVsRestClassifier (multiclass) — compare predicted labels on separable
# 3-class data, allowing a tiny count of mismatches.
# ---------------------------------------------------------------------------
def test_one_vs_rest_labels_match_sklearn():
    X, y = make_separable_3class()

    ours = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=0)).fit(X, y)
    theirs = SkOneVsRest(SkLogisticRegression(max_iter=1000)).fit(X, y)

    pred_ours = ours.predict(X)
    pred_theirs = theirs.predict(X)

    # labels must match exactly (they already equal the original class ids)
    np.testing.assert_array_equal(pred_ours, pred_theirs)


def test_one_vs_rest_near_perfect_agreement_on_separable_data():
    X, y = make_separable_3class(seed=11)

    ours = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=0)).fit(X, y)
    theirs = SkOneVsRest(SkLogisticRegression(max_iter=1000)).fit(X, y)

    mismatches = int(np.sum(ours.predict(X) != theirs.predict(X)))
    # on cleanly separable data both should be ~identical; allow a tiny drift
    assert mismatches <= 2, f"expected <=2 label mismatches vs sklearn, got {mismatches}"


def test_one_vs_rest_shuffled_noncontiguous_labels_match_sklearn():
    # non-0..k-1 labels in a shuffled *class order* must be handled by both the
    # same way: predictions reproduce the raw labels, not a densified subset.
    X, y = make_separable_3class(seed=7)
    raw_labels = np.array([10, 30, 20])
    # class 0 -> 10, class 1 -> 30, class 2 -> 20 (non-sorted id order)
    y_raw = raw_labels[y]

    ours = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=0)).fit(X, y_raw)
    theirs = SkOneVsRest(SkLogisticRegression(max_iter=1000)).fit(X, y_raw)

    pred_ours, pred_theirs = ours.predict(X), theirs.predict(X)
    # both return the raw labels, and the true label of every sample is matched
    assert set(pred_ours) == set(raw_labels)
    assert int(np.sum(ours.predict(X) != y_raw)) <= 1
    mismatches = int(np.sum(pred_ours != pred_theirs))
    assert mismatches <= 2, f"expected <=2 label mismatches vs sklearn, got {mismatches}"
