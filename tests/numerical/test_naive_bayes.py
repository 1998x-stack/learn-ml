"""Numerical tests for Naive Bayes estimators (s11 / v0.3.3)."""
import numpy as np
import pytest

from src.mini_sklearn.naive_bayes import GaussianNB


def make_separable_twofeature():
    """Two well-separated 2-feature, 2-class blobs (deterministic).

    Class 0 ~ N(mean=(1,1), sigma^2=(1,1)); class 1 ~ N(mean=(4,4), sigma^2=(1,1)).
    The per-feature variance is exactly 1, so the per-feature Gaussian
    log-pdf ``-0.5*log(2*pi*sigma^2) - 0.5*(x-mu)^2/sigma^2`` is hand-computable.
    """
    class0 = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])  # mean=(1,1)
    class1 = np.array([[3.0, 3.0], [5.0, 5.0], [3.0, 5.0], [5.0, 3.0]])  # mean=(4,4)
    X = np.vstack([class0, class1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _hand_gaussian_log_pdf(x, mu, sigma2):
    """Independent per-feature Gaussian log-pdf (the 'hand' reference)."""
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2


def test_gaussiannb_recovers_separable_classes():
    X, y = make_separable_twofeature()
    model = GaussianNB().fit(X, y)
    pred = model.predict(X)
    assert np.mean(pred == y) == 1.0  # separable: per-class means far apart


def test_predict_proba_rows_sum_to_one_and_in_unit_interval():
    X, y = make_separable_twofeature()
    model = GaussianNB().fit(X, y)
    P = model.predict_proba(X)
    assert P.shape == (X.shape[0], 2)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(P > 0.0) and np.all(P < 1.0)


def test_joint_log_likelihood_matches_hand_per_feature_gaussians():
    # "iris iris" close-features: two classes whose per-feature Gaussians are
    # close enough that only the summed log-probability decides.  We recompute
    # every per-feature Gaussian log-pdf by hand and check the argmax.
    X, y = make_separable_twofeature()
    model = GaussianNB().fit(X, y)
    probe = np.array([[1.0, 1.0], [4.0, 4.0]])  # one probe per class center

    theta = model.theta_          # (n_features, n_classes) means
    sigma2 = model.sigma_          # (n_features, n_classes) variances
    prior = model.class_log_prior_

    for px, expected in zip(probe, [0, 1]):
        log_lik = np.empty(2)
        for k in range(2):
            log_lik[k] = (
                _hand_gaussian_log_pdf(px[0], theta[0, k], sigma2[0, k])
                + _hand_gaussian_log_pdf(px[1], theta[1, k], sigma2[1, k])
            )
        log_post = log_lik + prior
        # framework must agree with the hand-rolled per-feature sum
        np.testing.assert_allclose(model.joint_log_likelihood(px[None, :])[0],
                                   log_post, rtol=1e-6)
        assert model.predict(px[None, :])[0] == expected  # argmax log-posterior


def test_predict_proba_wrong_n_features_raises():
    X, y = make_separable_twofeature()
    model = GaussianNB().fit(X, y)
    with pytest.raises(ValueError, match="different number of features"):
        model.predict_proba(np.zeros((3, 5)))


def test_predict_proba_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        GaussianNB().predict_proba(np.zeros((2, 2)))
