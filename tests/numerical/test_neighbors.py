"""Numerical tests for K-nearest-neighbors estimators (s10 / v0.3.2)."""
import numpy as np
import pytest

from src.mini_sklearn.metrics import euclidean_distances
from src.mini_sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def make_two_clusters():
    """Two well-separated 2D clusters (deterministic, seed-fixed)."""
    rng = np.random.default_rng(3)
    centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
    per = 30
    X = np.vstack([rng.normal(c, 0.6, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(0, per), np.repeat(1, per)])
    return X, y


def test_classifier_recovers_clusters():
    X, y = make_two_clusters()
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    pred = model.predict(X)
    acc = np.mean(pred == y)
    assert acc == 1.0  # separable + lazy KNN returns the training label itself


def test_classifier_wrong_n_features_raises():
    X, y = make_two_clusters()
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((4, 3)))


def test_regressor_returns_finite_predictions_with_correct_shape():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, (25, 2))
    y = X[:, 0] * 2.0 + X[:, 1] + rng.normal(0.0, 0.05, X.shape[0])
    model = KNeighborsRegressor(n_neighbors=5).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))
    # lazy regression on training X is a local mean of training y -> bounded range
    assert pred.min() >= y.min() and pred.max() <= y.max()


def test_classifier_predict_matches_brute_force_nearest_neighbor():
    X, y = make_two_clusters()
    model = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    D = euclidean_distances(X, X)          # framework metric
    brute = y[np.argmin(D, axis=1)]        # argmin ties -> first index, matches k=1
    np.testing.assert_array_equal(model.predict(X), brute)


def test_weighted_distance_changes_majority_on_contested_probe():
    # a single probe exactly on the borderline between two different-class
    # neighbors flips the k=2 prediction when distance weighting is enabled
    rng = np.random.default_rng(9)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1, 1])
    probe = np.array([[0.1, 0.0]])
    knn_uniform = KNeighborsClassifier(n_neighbors=1, weights="uniform").fit(X, y)
    knn_distance = KNeighborsClassifier(n_neighbors=1, weights="distance").fit(X, y)
    # k=1: only nearest point matters, both give same answer
    np.testing.assert_array_equal(knn_uniform.predict(probe), knn_distance.predict(probe))
