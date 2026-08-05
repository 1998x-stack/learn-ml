"""Estimator protocol tests for preprocessing transformers (s05 / v0.2.x)."""
import numpy as np
import pytest

from src.mini_sklearn.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer


def test_fit_returns_self():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc = StandardScaler()
    assert sc.fit(X) is sc
    mm = MinMaxScaler()
    assert mm.fit(X) is mm
    imp = SimpleImputer()
    assert imp.fit(X) is imp


def test_transform_before_fit_raises_value_error():
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="fitted"):
        StandardScaler().transform(X)
    with pytest.raises(ValueError, match="fitted"):
        MinMaxScaler().transform(X)
    with pytest.raises(ValueError, match="fitted"):
        SimpleImputer().transform(np.array([[1.0, np.nan]]))


def test_get_params_returns_only_hyperparameters():
    sc = StandardScaler()
    assert sc.get_params() == {"with_mean": True, "with_std": True}
    mm = MinMaxScaler(with_min=0.0, with_max=1.0)
    assert mm.get_params() == {"with_min": 0.0, "with_max": 1.0}
    imp = SimpleImputer()
    assert imp.get_params() == {"strategy": "mean"}


def test_fit_transform_via_transformer_mixin():
    X = np.array([[0.0], [2.0], [4.0]])
    Xt = StandardScaler().fit_transform(X)
    assert isinstance(Xt, np.ndarray)
    np.testing.assert_allclose(Xt.mean(), 0.0, atol=1e-12)


def test_learned_attrs_have_trailing_underscore_not_in_params():
    sc = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert hasattr(sc, "mean_") and hasattr(sc, "scale_")
    assert hasattr(sc, "n_features_in_")
    assert "mean_" not in sc.get_params()
