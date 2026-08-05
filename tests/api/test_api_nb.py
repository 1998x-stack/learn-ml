"""Estimator-protocol tests for Naive Bayes estimators (s11 / v0.3.3)."""
import numpy as np
import pytest

from src.mini_sklearn.base import clone
from src.mini_sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

X = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
y = np.array([0, 0, 1, 1, 1])


def test_fit_returns_self():
    model = GaussianNB()
    assert model.fit(X, y) is model


def test_predict_proba_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        GaussianNB().predict_proba(X)
    with pytest.raises(ValueError, match="fitted"):
        GaussianNB().predict(X)
    with pytest.raises(ValueError, match="fitted"):
        MultinomialNB().predict_proba(X)
    with pytest.raises(ValueError, match="fitted"):
        BernoulliNB().predict_proba(X)


def test_get_params_hyper_only():
    model = GaussianNB(var_smoothing=1e-7)
    assert model.get_params() == {"var_smoothing": 1e-7}
    model.fit(X, y)
    assert "theta_" not in model.get_params()
    assert "sigma_" not in model.get_params()
    assert "class_prior_" not in model.get_params()
    assert "classes_" not in model.get_params()


def test_multinomial_bernoulli_params_hyper_only():
    assert MultinomialNB().get_params() == {"alpha": 1.0}
    assert BernoulliNB().get_params() == {"alpha": 1.0}
    assert MultinomialNB(alpha=0.5).get_params() == {"alpha": 0.5}


def test_set_params_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Invalid parameter"):
        GaussianNB().set_params(bogus=1)


def test_clone_reproduces_hyperparameters():
    for model in (GaussianNB(var_smoothing=1e-5), MultinomialNB(alpha=0.3),
                  BernoulliNB(alpha=2.0)):
        cloned = clone(model)
        assert cloned is not model
        assert cloned.get_params() == model.get_params()


def test_learned_state_stored_and_classifiers():
    model = GaussianNB().fit(X, y)
    assert hasattr(model, "theta_")      # (n_features, n_classes) means
    assert hasattr(model, "sigma_")      # variances incl. epsilon guard
    assert hasattr(model, "class_count_")
    assert hasattr(model, "class_prior_")
    assert hasattr(model, "class_log_prior_")
    assert hasattr(model, "epsilon_")
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_n_features_in_set_after_fit():
    model = GaussianNB().fit(X, y)
    assert model.n_features_in_ == X.shape[1]


def test_is_classifier():
    assert GaussianNB()._estimator_type == "classifier"
    assert MultinomialNB()._estimator_type == "classifier"
    assert BernoulliNB()._estimator_type == "classifier"
