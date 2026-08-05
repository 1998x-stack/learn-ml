"""Estimator protocol tests for linear regression (s06 / v0.2.3)."""
import numpy as np
import pytest

from src.mini_sklearn.linear_model import LinearRegression, Ridge


def test_fit_returns_self():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    lr = LinearRegression()
    assert lr.fit(X, y) is lr
    rg = Ridge()
    assert rg.fit(X, y) is rg


def test_predict_before_fit_raises_value_error():
    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="fitted"):
        LinearRegression().predict(X)
    with pytest.raises(ValueError, match="fitted"):
        Ridge().predict(X)


def test_get_params_hyper_only():
    lr = LinearRegression()
    assert lr.get_params() == {"fit_intercept": True}
    rg = Ridge()
    assert rg.get_params() == {"alpha": 1.0, "fit_intercept": True}
    # learned attrs stay out of get_params
    lr.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    assert "coef_" not in lr.get_params()
    assert "intercept_" not in lr.get_params()


def test_set_params_unknown_raises_value_error():
    lr = LinearRegression()
    with pytest.raises(ValueError, match="Invalid parameter"):
        lr.set_params(not_a_param=1)
    rg = Ridge()
    with pytest.raises(ValueError, match="Invalid parameter"):
        rg.set_params(alpha=None, bogus=True)


def test_set_params_updates_and_keeps_self():
    lr = LinearRegression(fit_intercept=True)
    out = lr.set_params(fit_intercept=False)
    assert out is lr
    assert lr.get_params() == {"fit_intercept": False}

    rg = Ridge(alpha=1.0)
    rg.set_params(alpha=0.1)
    assert rg.get_params() == {"alpha": 0.1, "fit_intercept": True}


def test_learned_attrs_have_trailing_underscore():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    lr = LinearRegression().fit(X, y)
    assert hasattr(lr, "coef_") and hasattr(lr, "intercept_")
    assert hasattr(lr, "n_features_in_")
    rg = Ridge().fit(X, y)
    assert hasattr(rg, "coef_") and hasattr(rg, "intercept_")
