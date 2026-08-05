"""Estimator protocol tests for binary logistic regression (s08 / v0.2.6-7)."""
import numpy as np
import pytest

from src.mini_sklearn.linear_model import LogisticRegression

X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y = np.array([0, 1, 1, 0, 1])


def test_fit_returns_self():
    model = LogisticRegression()
    assert model.fit(X, y) is model


def test_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        LogisticRegression().predict(X)
    with pytest.raises(ValueError, match="fitted"):
        LogisticRegression().predict_proba(X)
    with pytest.raises(ValueError, match="fitted"):
        LogisticRegression().decision_function(X)


def test_get_params_hyper_only():
    model = LogisticRegression()
    assert model.get_params() == {
        "learning_rate": 0.1,
        "max_iter": 1000,
        "random_state": 0,
        "fit_intercept": True,
    }
    model.fit(X, y)
    assert "coef_" not in model.get_params()
    assert "intercept_" not in model.get_params()
    assert "classes_" not in model.get_params()
    assert "loss_history_" not in model.get_params()


def test_set_params_unknown_raises_value_error():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_param=1)
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(learning_rate=0.5, bogus=True)


def test_set_params_updates_and_keeps_self():
    model = LogisticRegression(learning_rate=0.1)
    out = model.set_params(learning_rate=0.5)
    assert out is model
    assert model.get_params()["learning_rate"] == 0.5


def test_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(7)
    Xn = rng.standard_normal((120, 3))
    centers = rng.normal(size=(2, 3))
    yn = rng.integers(0, 2, size=120)
    Xn = Xn + centers[yn]
    a = LogisticRegression(random_state=0, max_iter=500).fit(Xn, yn)
    b = LogisticRegression(random_state=0, max_iter=500).fit(Xn, yn)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.intercept_, b.intercept_)
    np.testing.assert_array_equal(a.classes_, b.classes_)


def test_learned_attrs_have_trailing_underscore():
    model = LogisticRegression().fit(X, y)
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "classes_")
    assert hasattr(model, "n_features_in_")
    assert hasattr(model, "loss_history_")
    assert hasattr(model, "n_iter_")


def test_is_classifier():
    assert LogisticRegression()._estimator_type == "classifier"