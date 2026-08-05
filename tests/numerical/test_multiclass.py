"""Numerical tests for the multiclass one-vs-rest strategy (s09 / v0.2.8)."""
import numpy as np
import pytest

from src.mini_sklearn.linear_model import LogisticRegression
from src.mini_sklearn.multiclass import LabelEncoder, OneVsRestClassifier


def make_3class():
    """Three well-separated 2D Gaussian blobs (deterministic, seed-fixed).

    Centers form an equilateral-ish triangle far apart so a linear
    one-vs-rest boundary cleanly separates every class.
    """
    rng = np.random.default_rng(7)
    centers = np.array([[-4.0, -4.0], [4.0, -4.0], [0.0, 4.0]])
    per = 50
    X = np.vstack([rng.normal(c, 0.5, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(i, per) for i in range(3)])
    return X, y


def test_one_vs_rest_produces_exactly_three_classes():
    X, y = make_3class()
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    pred = model.predict(X)
    assert len(np.unique(pred)) == 3
    np.testing.assert_array_equal(np.unique(pred), np.unique(y))


def test_training_accuracy_high_on_separable_3class():
    X, y = make_3class()
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.8


def test_predict_proba_shape_and_normalised_rows():
    X, y = make_3class()
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    P = model.predict_proba(X)
    assert P.shape == (X.shape[0], 3)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_argmax_of_scores_matches_predict():
    X, y = make_3class()
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    scores = np.column_stack(
        [est.predict_proba(X)[:, 1] for est in model.estimators_]
    )
    expected = model.classes_[np.argmax(scores, axis=1)]
    np.testing.assert_array_equal(model.predict(X), expected)


def test_number_of_fitted_clones_equals_n_classes():
    X, y = make_3class()
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    assert len(model.estimators_) == 3
    assert model.n_features_in_ == X.shape[1]


def test_label_encoder_round_trip_preserves_identity():
    y = np.array([2, 0, 2, 1, 0, 2, 1, 0])
    enc = LabelEncoder().fit(y)
    encoded = enc.transform(y)
    np.testing.assert_array_equal(enc.inverse_transform(encoded), y)


def test_label_encoder_transform_preserves_fit_class_order():
    y = np.array(["b", "a", "c", "b", "a"])
    enc = LabelEncoder().fit(y)
    # classes_ from fit -> the order transform should map to
    np.testing.assert_array_equal(enc.classes_, np.array(["a", "b", "c"]))
    np.testing.assert_array_equal(enc.transform(y), np.array([1, 0, 2, 1, 0]))


def test_label_encoder_rows_sum_and_range_preserved_for_strange_labels():
    # uneven, non-0..k-1 integer labels must densify to 0..k-1
    y = np.array([10, 200, 3, 200, 3, 10])
    enc = LabelEncoder().fit(y)
    encoded = enc.transform(y)
    assert encoded.min() == 0 and encoded.max() == 2
    np.testing.assert_array_equal(enc.inverse_transform(encoded), y)


def test_unknown_label_in_transform_raises():
    enc = LabelEncoder().fit(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="not seen"):
        enc.transform(np.array([1, 9]))
