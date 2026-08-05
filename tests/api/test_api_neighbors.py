"""Estimator-protocol tests for K-nearest-neighbors estimators (s10 / v0.3.2)."""
import numpy as np
import pytest

from src.mini_sklearn.base import clone
from src.mini_sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
y_class = np.array([0, 0, 1, 1, 1])
y_reg = np.array([0.5, 1.0, 2.0, 3.0, 4.0])


def test_fit_returns_self():
    assert KNeighborsClassifier(n_neighbors=2).fit(X, y_class) is not None
    a = KNeighborsClassifier(n_neighbors=2)
    assert a.fit(X, y_class) is a


def test_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        KNeighborsClassifier().predict(X)


def test_regressor_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        KNeighborsRegressor().predict(X)


def test_get_params_returns_only_hyperparams():
    model = KNeighborsClassifier(n_neighbors=3, metric="manhattan", weights="uniform")
    assert model.get_params() == {
        "n_neighbors": 3,
        "metric": "manhattan",
        "weights": "uniform",
    }


def test_get_params_no_learned_state_surfaces():
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y_class)
    params = model.get_params()
    assert "_fit_X" not in params
    assert "classes_" not in params
    assert "n_features_in_" not in params


def test_n_features_in_set_after_fit():
    model = KNeighborsClassifier().fit(X, y_class)
    assert model.n_features_in_ == X.shape[1]


def test_clone_reproduces_hyperparameters():
    model = KNeighborsRegressor(n_neighbors=4, metric="manhattan")
    cloned = clone(model)
    assert cloned is not model
    assert cloned.get_params() == model.get_params()


def test_stats_learning_state_stored():
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y_class)
    assert hasattr(model, "_fit_X")
    assert hasattr(model, "_fit_y")
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_is_classifier_and_regressor():
    assert KNeighborsClassifier()._estimator_type == "classifier"
    assert KNeighborsRegressor()._estimator_type == "regressor"
