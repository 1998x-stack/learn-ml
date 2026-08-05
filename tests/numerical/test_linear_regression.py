"""Numerical tests for linear regression estimators (s06 / v0.2.3)."""
import numpy as np
import pytest

from src.mini_sklearn.linear_model import LinearRegression, Ridge


def test_linear_regression_hand_computable_1d():
    # y = 1.0 * x + 1.0 exactly (slope 1, intercept 1)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, [1.0], atol=1e-8)
    np.testing.assert_allclose(model.intercept_, 1.0, atol=1e-8)
    # perfect fit -> R^2 == 1
    y_pred = model.predict(X)
    np.testing.assert_allclose(y_pred, y, atol=1e-8)


def test_linear_regression_no_intercept_forces_origin():
    # y = 2 * x through origin; without intercept the line must pass (0,0)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0])
    model = LinearRegression(fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(model.intercept_, 0.0, atol=1e-8)
    np.testing.assert_allclose(model.coef_, [2.0], atol=1e-8)


def test_linear_regression_multi_feature():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 3.0
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, true_w, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, 3.0, atol=1e-8)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)


def test_linear_regression_sets_n_features_in():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = LinearRegression().fit(X, y)
    assert model.n_features_in_ == 2


def test_linear_regression_get_params_hyper_only():
    model = LinearRegression()
    assert model.get_params() == {"fit_intercept": True}
    model = LinearRegression(fit_intercept=False)
    assert model.get_params() == {"fit_intercept": False}


def test_ridge_fits_noisy_data_and_predict_shape():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 2))
    true_w = np.array([1.5, -2.0])
    y = X @ true_w + 0.5 + 0.1 * rng.standard_normal(300)
    model = Ridge(alpha=0.01).fit(X, y)
    np.testing.assert_allclose(model.coef_, true_w, atol=0.2)
    np.testing.assert_allclose(model.intercept_, 0.5, atol=0.2)
    y_pred = model.predict(X)
    assert y_pred.shape == (300,)


def test_ridge_stores_n_features_in():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = Ridge(alpha=0.5).fit(X, y)
    assert model.n_features_in_ == 2


def test_ridge_get_params_hyper_only():
    model = Ridge()
    assert model.get_params() == {"alpha": 1.0, "fit_intercept": True}
    model = Ridge(alpha=0.5, fit_intercept=False)
    assert model.get_params() == {"alpha": 0.5, "fit_intercept": False}


def test_ridge_alpha_regularizes_ill_posed():
    # two identical columns -> X^T X is singular; ridge still solves it
    x = np.linspace(0, 1, 20)
    X = np.column_stack([x, x])  # collinear duplicated feature
    y = 2.0 * x + 1.0
    model = Ridge(alpha=1.0).fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (20,)
    # predictions stay finite and reasonably close to targets
    np.testing.assert_allclose(y_pred, y, atol=0.5)


def test_predict_rejects_wrong_feature_count():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[0.0], [1.0]]))
