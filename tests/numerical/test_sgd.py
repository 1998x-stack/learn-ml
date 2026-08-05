"""Numerical tests for iterative optimization estimators (s07 / v0.2.4-5)."""
import numpy as np
import pytest

from src.mini_sklearn.linear_model import GDRegressor, SGDRegressor


def make_1d_dataset(rng):
    """y = 3 * X + 2 (+ small noise), single feature, 50 samples."""
    X = np.linspace(-2, 2, 50)[:, None]
    y = 3.0 * X[:, 0] + 2.0 + 0.1 * rng.standard_normal(50)
    return X, y


def test_gd_loss_history_is_decreasing():
    rng = np.random.default_rng(0)
    X, y = make_1d_dataset(rng)
    model = GDRegressor(learning_rate=0.05, max_iter=2000).fit(X, y)
    hist = model.loss_history_
    # strictly decreasing: no gradient step ever increases the MSE
    assert all(b <= a for a, b in zip(hist, hist[1:]))
    # and it actually got better than the initial guess
    assert hist[-1] < hist[0]


def test_gd_recovers_true_params():
    rng = np.random.default_rng(0)
    X, y = make_1d_dataset(rng)
    model = GDRegressor(learning_rate=0.05, max_iter=2000).fit(X, y)
    np.testing.assert_allclose(model.coef_[0], 3.0, atol=0.2)
    np.testing.assert_allclose(model.intercept_, 2.0, atol=0.2)
    assert model.n_iter_ > 0
    assert model.converged_ is True


def test_gd_predict_shape():
    rng = np.random.default_rng(0)
    X, y = make_1d_dataset(rng)
    model = GDRegressor(learning_rate=0.05, max_iter=2000).fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (50,)


def test_sgd_approaches_truth():
    rng = np.random.default_rng(1)
    X, y = make_1d_dataset(rng)
    model = SGDRegressor(learning_rate=0.01, max_iter=2000, random_state=0).fit(X, y)
    # stochastic noise -> looser tolerance than full-batch GD
    np.testing.assert_allclose(model.coef_[0], 3.0, atol=0.6)
    np.testing.assert_allclose(model.intercept_, 2.0, atol=0.6)
    assert model.n_iter_ > 0
    assert len(model.loss_history_) == model.n_iter_ + 1  # initial + one per epoch


def test_gd_get_params_hyper_only():
    model = GDRegressor()
    assert model.get_params() == {
        "learning_rate": 0.01,
        "max_iter": 1000,
        "tol": 1e-4,
        "fit_intercept": True,
    }


def test_sgd_get_params_hyper_only():
    model = SGDRegressor()
    assert model.get_params() == {
        "learning_rate": 0.01,
        "max_iter": 1000,
        "random_state": 0,
        "fit_intercept": True,
    }
