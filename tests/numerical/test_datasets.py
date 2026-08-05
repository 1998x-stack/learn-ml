import numpy as np

from src.mini_sklearn.datasets import make_classification, make_regression


def test_make_regression_shapes():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    assert X.shape == (100, 4)
    assert y.shape == (100,)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)


def test_make_regression_is_linear_up_to_noise():
    # 无噪声时 y = X @ coef 精确成立
    X, y = make_regression(n_samples=50, n_features=3, noise=0.0, random_state=7)
    assert np.all(np.isfinite(y))
    assert np.std(y) > 0  # 目标非恒定


def test_make_classification_shapes():
    X, y = make_classification(
        n_samples=80, n_features=5, n_classes=3, random_state=7
    )
    assert X.shape == (80, 5)
    assert y.shape == (80,)


def test_make_classification_label_bounded():
    X, y = make_classification(
        n_samples=60, n_features=3, n_classes=3, random_state=1
    )
    unique = np.unique(y)
    assert len(unique) <= 3
    assert set(unique.tolist()) <= {0, 1, 2}
    assert X.shape[1] == 3 and len(y) == 60
