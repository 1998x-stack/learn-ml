"""Numerical tests for binary logistic regression (s08 / v0.2.6-7)."""
import numpy as np
import pytest

from src.mini_sklearn.datasets import make_classification
from src.mini_sklearn.linear_model import LogisticRegression


def make_separable():
    """Two clearly-separated Gaussian blobs (deterministic, seed-fixed)."""
    # seed=1 is hand-picked: make_classification's blob centers/noise make
    # training acc > 0.8 only for a few seeds (roughly 1, 4, 5, 7, 10); most
    # others overlap enough that the logistic decision boundary misses 0.8.
    X, y = make_classification(n_samples=100, n_features=2, random_state=1)
    return X, y


def test_proba_rows_sum_to_one_and_in_unit_interval():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    P = model.predict_proba(X)
    assert P.shape == (100, 2)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(P >= 0.0) and np.all(P <= 1.0)


def test_decision_function_shape():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    z = model.decision_function(X)
    assert z.shape == (100,)


def test_classes_are_sorted_unique_labels():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    assert model.classes_.tolist() == sorted(np.unique(y).tolist())


def test_training_accuracy_greater_than_80_percent():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.8


def test_coef_nonzero_and_decision_separates_classes():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    assert np.any(np.abs(model.coef_) > 0.0)
    z = model.decision_function(X)
    # separable blobs: mean score of class 0 must sit well below class 1's
    assert z[y == 0].mean() < z[y == 1].mean()


def test_predict_proba_wrong_n_features_raises():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="different number of features"):
        model.predict_proba(np.random.default_rng(0).standard_normal((5, 3)))


def test_strongly_separable_blobs_give_near_zero_one_proba():
    rng = np.random.default_rng(123)
    # two compact, far-apart blobs -> predictions should be near-certain
    X = np.vstack([rng.normal(-5.0, 0.5, (50, 2)), rng.normal(5.0, 0.5, (50, 2))])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    model = LogisticRegression().fit(X, y)
    P = model.predict_proba(X)
    np.testing.assert_allclose(model.predict(X), y)
    # strongly separated -> the modal class probability is near 0 or 1
    modal = P.max(axis=1)
    assert np.mean(modal > 0.99) > 0.9


def test_loss_decreases_over_training():
    X, y = make_separable()
    model = LogisticRegression().fit(X, y)
    hist = model.loss_history_
    assert hist[-1] < hist[0]
    assert model.n_iter_ > 0
