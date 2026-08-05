"""Numerical tests for preprocessing transformers (s05 / v0.2.x)."""
import numpy as np
import pytest

from src.mini_sklearn.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer


def test_standard_scaler_stores_train_stats():
    X = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, X.mean(axis=0))
    np.testing.assert_allclose(sc.scale_, X.std(axis=0, ddof=0))
    assert sc.n_features_in_ == 2


def test_standard_scaler_transform_zero_mean_unit_var():
    X = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    sc = StandardScaler().fit(X)
    Xt = sc.transform(X)
    np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xt.std(axis=0, ddof=0), 1.0, atol=1e-12)


def test_standard_scaler_inverse_roundtrip():
    X = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    Xt = StandardScaler().fit(X).transform(X)
    np.testing.assert_allclose(StandardScaler().fit(X).inverse_transform(Xt), X, atol=1e-12)


def test_standard_scaler_transform_unseen_uses_train_stats():
    X_train = np.array([[0.0], [2.0], [4.0]])
    X_test = np.array([[1.0], [3.0]])
    sc = StandardScaler().fit(X_train)
    Xt = sc.transform(X_test)
    # must use the TRAIN mean/scale, not the test's own stats
    expected = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
    np.testing.assert_allclose(Xt, expected)


def test_standard_scaler_raises_when_not_fitted():
    with pytest.raises(ValueError, match="fitted"):
        StandardScaler().transform([[1.0, 2.0]])


def test_standard_scaler_rejects_wrong_feature_count():
    sc = StandardScaler().fit(np.array([[0.0, 1.0], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="features"):
        sc.transform(np.array([[1.0], [2.0]]))


def test_minmax_scaler_range():
    X = np.array([[-1.0, 2.0], [0.0, 3.0], [1.0, 4.0]])
    mm = MinMaxScaler().fit(X)
    np.testing.assert_allclose(mm.min_, X.min(axis=0))
    np.testing.assert_allclose(mm.scale_, X.max(axis=0) - X.min(axis=0))
    Xt = mm.transform(X)
    np.testing.assert_allclose(Xt.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xt.max(axis=0), 1.0, atol=1e-12)


def test_min_max_scaler_inverse_roundtrip():
    X = np.array([[-1.0, 2.0], [0.0, 3.0], [1.0, 4.0]])
    mm = MinMaxScaler().fit(X)
    Xt = mm.transform(X)
    np.testing.assert_allclose(mm.inverse_transform(Xt), X, atol=1e-12)


def test_min_max_scaler_zero_range_guarded():
    X = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
    mm = MinMaxScaler().fit(X)
    np.testing.assert_allclose(mm.scale_, [1.0, 2.0])
    Xt = mm.transform(X)
    np.testing.assert_allclose(Xt[:, 0], 0.0, atol=1e-12)


def test_simple_imputer_strategy_mean():
    X = np.array([[1.0, np.nan], [2.0, 6.0], [np.nan, 8.0]])
    imp = SimpleImputer(strategy="mean").fit(X)
    np.testing.assert_allclose(imp.statistics_, [1.5, 7.0])
    X_filled = imp.transform(X)
    assert not np.isnan(X_filled).any()
    np.testing.assert_allclose(X_filled[:, 1], [7.0, 6.0, 8.0])
    np.testing.assert_allclose(X_filled[:, 0], [1.0, 2.0, 1.5])


def test_simple_imputer_strategy_median():
    X = np.array([[1.0, np.nan], [3.0, 7.0], [2.0, 8.0]])
    imp = SimpleImputer(strategy="median").fit(X)
    np.testing.assert_allclose(imp.statistics_, [2.0, 7.5])


def test_simple_imputer_raises_when_not_fitted():
    with pytest.raises(ValueError, match="fitted"):
        SimpleImputer().transform(np.array([[1.0, np.nan]]))
