"""Estimator-protocol / meta-estimator tests for OneVsRestClassifier (s09 / v0.2.8)."""
import numpy as np
import pytest

from src.mini_sklearn.base import clone
from src.mini_sklearn.linear_model import LogisticRegression
from src.mini_sklearn.multiclass import LabelEncoder, OneVsRestClassifier

X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0]])
y = np.array([0, 1, 2, 0, 1])


def test_fit_returns_self():
    model = OneVsRestClassifier(LogisticRegression())
    assert model.fit(X, y) is model


def test_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        OneVsRestClassifier(LogisticRegression()).predict(X)


def test_predict_proba_before_fit_raises_value_error():
    # protocol parity: predict AND predict_proba must both be ValueErrors pre-fit
    with pytest.raises(ValueError, match="fitted"):
        OneVsRestClassifier(LogisticRegression()).predict_proba(X)


def test_get_params_returns_estimator():
    base = LogisticRegression(learning_rate=0.3, max_iter=500)
    model = OneVsRestClassifier(base)
    params = model.get_params(deep=True)
    # teaching choice: only the wrapped estimator is surfaced, so the
    # framework clone() (which does __class__(**get_params())) keeps working.
    assert params == {"estimator": base}


def test_get_params_has_no_learned_state():
    model = OneVsRestClassifier(LogisticRegression()).fit(X, y)
    params = model.get_params(deep=True)
    assert "estimators_" not in params
    assert "classes_" not in params
    assert "n_features_in_" not in params


def test_clones_are_independent_instances():
    model = OneVsRestClassifier(LogisticRegression()).fit(X, y)
    # Every per-class clone must be a distinct, separately-fitted model, not
    # the same object reused, and none may be the original estimator instance.
    assert model.estimator is not model.estimators_[0]
    clone_ids = {id(e) for e in model.estimators_}
    assert len(clone_ids) == len(model.estimators_)
    # each clone exposes learned weights -> it was genuinely fit
    for est in model.estimators_:
        assert hasattr(est, "coef_")


def test_clone_meta_estimator_reproduces_hyperparameters():
    model = OneVsRestClassifier(LogisticRegression(learning_rate=0.2))
    cloned = clone(model)
    assert cloned.estimator.learning_rate == 0.2
    assert cloned is not model


def test_set_params_replaces_wrapped_estimator():
    model = OneVsRestClassifier(LogisticRegression())
    replacement = LogisticRegression(learning_rate=0.9)
    model.set_params(estimator=replacement)
    assert model.estimator is replacement


def test_set_params_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Invalid parameter"):
        OneVsRestClassifier(LogisticRegression()).set_params(bad_param=1)


def test_is_classifier():
    assert OneVsRestClassifier(LogisticRegression())._estimator_type == "classifier"


def test_label_encoder_transform_before_fit_raises():
    with pytest.raises(ValueError, match="fitted"):
        LabelEncoder().transform(np.array([1, 2]))
    with pytest.raises(ValueError, match="fitted"):
        LabelEncoder().inverse_transform(np.array([0]))


def test_label_encoder_fit_returns_self():
    enc = LabelEncoder()
    assert enc.fit(np.array([1, 2, 1])) is enc
