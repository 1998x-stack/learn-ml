"""Estimator-protocol tests for decision trees (s12 / v0.3.4-5)."""
import numpy as np
import pytest

from src.mini_sklearn.base import clone
from src.mini_sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X = np.array(
    [[0.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
)
y_class = np.array([0, 0, 1, 1, 1])
y_reg = np.array([0.5, 1.0, 2.0, 3.0, 4.0])


def test_fit_returns_self():
    model = DecisionTreeClassifier()
    assert model.fit(X, y_class) is model
    reg = DecisionTreeRegressor()
    assert reg.fit(X, y_reg) is reg


def test_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        DecisionTreeClassifier().predict(X)
    with pytest.raises(ValueError, match="fitted"):
        DecisionTreeRegressor().predict(X)


def test_get_params_hyper_only():
    model = DecisionTreeClassifier(
        criterion="gini", max_depth=4, min_samples_split=3, random_state=1
    )
    assert model.get_params() == {
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_split": 3,
        "random_state": 1,
    }
    # learned state must not leak into the param table
    model.fit(X, y_class)
    params = model.get_params()
    assert "tree_" not in params
    assert "classes_" not in params
    assert "n_features_in_" not in params


def test_regressor_get_params_hyper_only():
    model = DecisionTreeRegressor(criterion="squared_error", max_depth=2,
                                  min_samples_split=2, random_state=0)
    assert model.get_params() == {
        "criterion": "squared_error",
        "max_depth": 2,
        "min_samples_split": 2,
        "random_state": 0,
    }


def test_n_features_in_set_after_fit():
    model = DecisionTreeClassifier().fit(X, y_class)
    assert model.n_features_in_ == X.shape[1]
    reg = DecisionTreeRegressor().fit(X, y_reg)
    assert reg.n_features_in_ == X.shape[1]


def test_clone_reproduces_hyperparameters():
    model = DecisionTreeRegressor(criterion="squared_error", max_depth=3,
                                  min_samples_split=4, random_state=7)
    cloned = clone(model)
    assert cloned is not model
    assert cloned.get_params() == model.get_params()


def test_learned_state_stored():
    model = DecisionTreeClassifier().fit(X, y_class)
    assert hasattr(model, "classes_")
    assert hasattr(model, "n_features_in_")
    assert hasattr(model, "tree_")
    assert hasattr(model, "feature_importances_")
    assert model.feature_importances_.shape[0] == X.shape[1]


def test_classifier_and_regressor_estimator_type():
    assert DecisionTreeClassifier()._estimator_type == "classifier"
    assert DecisionTreeRegressor()._estimator_type == "regressor"


def test_fit_accepts_sample_weight_for_future_adaboost():
    model = DecisionTreeClassifier(random_state=0).fit(X, y_class, sample_weight=np.ones(5))
    pred = model.predict(X)
    assert pred.shape == (5,)
