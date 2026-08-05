"""Estimator protocol tests for optimization estimators (s07 / v0.2.4-5)."""
import numpy as np
import pytest

from src.mini_sklearn.linear_model import GDRegressor, SGDRegressor


X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
y = np.array([1.0, 2.0, 3.0, 4.0])


def test_fit_returns_self():
    gd = GDRegressor()
    assert gd.fit(X, y) is gd
    sgd = SGDRegressor()
    assert sgd.fit(X, y) is sgd


def test_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        GDRegressor().predict(X)
    with pytest.raises(ValueError, match="fitted"):
        SGDRegressor().predict(X)


def test_get_params_hyper_only():
    gd = GDRegressor()
    assert gd.get_params() == {
        "learning_rate": 0.01,
        "max_iter": 1000,
        "tol": 1e-4,
        "fit_intercept": True,
    }
    sgd = SGDRegressor()
    assert sgd.get_params() == {
        "learning_rate": 0.01,
        "max_iter": 1000,
        "random_state": 0,
        "fit_intercept": True,
    }
    # learned attrs stay out of get_params
    gd.fit(X, y)
    assert "coef_" not in gd.get_params()
    assert "intercept_" not in gd.get_params()
    assert "loss_history_" not in gd.get_params()


def test_set_params_unknown_raises_value_error():
    gd = GDRegressor()
    with pytest.raises(ValueError, match="Invalid parameter"):
        gd.set_params(not_a_param=1)
    sgd = SGDRegressor()
    with pytest.raises(ValueError, match="Invalid parameter"):
        sgd.set_params(random_state=None, bogus=True)


def test_set_params_updates_and_keeps_self():
    gd = GDRegressor(learning_rate=0.01)
    out = gd.set_params(learning_rate=0.5)
    assert out is gd
    assert gd.get_params()["learning_rate"] == 0.5


def test_sgd_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(7)
    Xn = rng.standard_normal((100, 3))
    yn = Xn @ np.array([2.0, -1.0, 0.5]) + 1.0
    a = SGDRegressor(random_state=0, max_iter=500).fit(Xn, yn)
    b = SGDRegressor(random_state=0, max_iter=500).fit(Xn, yn)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.intercept_, b.intercept_)


def test_learned_attrs_have_trailing_underscore():
    gd = GDRegressor().fit(X, y)
    assert hasattr(gd, "coef_") and hasattr(gd, "intercept_")
    assert hasattr(gd, "n_features_in_") and hasattr(gd, "loss_history_")
    sgd = SGDRegressor().fit(X, y)
    assert hasattr(sgd, "coef_") and hasattr(sgd, "intercept_")
