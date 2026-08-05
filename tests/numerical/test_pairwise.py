"""Numerical tests for pairwise distance/similarity metrics (s10 / v0.3.1)."""
import numpy as np

from src.mini_sklearn.metrics import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)


def test_euclidean_distances_hand_computable():
    # dist((0,0),(3,4)) = 5 ; diagonal is 0
    D = euclidean_distances([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_array_equal(D, [[0.0, 5.0], [5.0, 0.0]])


def test_manhattan_distances_hand_computable():
    # |0-3| + |0-4| = 7 ; diagonal is 0
    D = manhattan_distances([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_array_equal(D, [[0.0, 7.0], [7.0, 0.0]])


def test_cosine_similarity_self_diagonal_ones():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    S = cosine_similarity(X)
    np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-12)


def test_cosine_similarity_orthogonal_rows_zero():
    S = cosine_similarity([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(S[0, 1], 0.0, atol=1e-12)


def test_self_distance_equals_explicit_pair():
    # round-trip: euclidean_distances(X) must equal euclidean_distances(X, X)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    np.testing.assert_allclose(euclidean_distances(X), euclidean_distances(X, X))


def test_shape_is_nx_by_ny():
    X = np.zeros((4, 2))
    Y = np.zeros((7, 2))
    assert euclidean_distances(X, Y).shape == (4, 7)
    assert manhattan_distances(X, Y).shape == (4, 7)
    assert cosine_similarity(X, Y).shape == (4, 7)


def test_cosine_normalizes_rows_before_dot():
    # scaling just row norms, cosine unchanged
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0]])
    # hand: cos between (1,2) and (2,0) = (2)/ (sqrt5 * 2) = 1/sqrt5
    S = cosine_similarity(a, b)
    np.testing.assert_allclose(S[0, 0], 1.0 / np.sqrt(5.0), atol=1e-12)
