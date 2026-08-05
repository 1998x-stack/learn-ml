import numpy as np
import pytest
from src.mini_sklearn.utils.validation import (
    check_array,
    check_X_y,
    check_is_fitted,
    check_consistent_length,
    check_random_state,
)

def test_check_array_accepts_2d_float():
    out = check_array([[1.0, 2.0], [3.0, 4.0]])
    assert out.ndim == 2

def test_check_array_rejects_1d():
    with pytest.raises(ValueError, match="2d"):
        check_array([1.0, 2.0, 3.0])

def test_check_array_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        check_array([[1.0, np.nan]])

def test_check_X_y_feature_count_error_later():
    X = np.ones((3, 2)); y = np.ones(4)
    with pytest.raises(ValueError):
        check_X_y(X, y)

def test_check_is_fitted_raises_when_missing():
    class Dummy:
        pass
    with pytest.raises(ValueError, match="fitted"):
        check_is_fitted(Dummy(), "coef_")

def test_check_consistent_length_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        check_consistent_length([1, 2], [1])

def test_random_state_accepts_int_and_rng():
    assert isinstance(check_random_state(42), np.random.Generator)
    assert isinstance(check_random_state(np.random.default_rng(0)), np.random.Generator)
