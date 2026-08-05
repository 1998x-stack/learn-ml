"""s03 Data Contract — 先校验数据，再学习参数 (v0.1.4).

Run:    python s03_data_validation/code.py
"""
# --- v0.1.4: 输入与状态校验 check_array / check_X_y / check_is_fitted ---
from __future__ import annotations

import numpy as np


def _check_array(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got ndim={X.ndim}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinity; not allowed by default")
    return X


def _check_X_y(X, y):
    X = _check_array(X)
    y = np.asarray(y)
    if len(y) != X.shape[0]:
        raise ValueError(f"inconsistent lengths: X={X.shape[0]}, y={len(y)}")
    return X, y


def _check_is_fitted(model, attribute):
    if not hasattr(model, attribute):
        raise ValueError(f"This {type(model).__name__} is not fitted yet.")


class MeanRegressor:
    def fit(self, X, y):
        X, y = _check_X_y(X, y)
        self.constant_ = float(y.mean())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        _check_is_fitted(self, "constant_")
        X = _check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from fitted data")
        return np.full(X.shape[0], self.constant_)


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((15, 3))
    y = 2.0 + rng.standard_normal(15)

    model = MeanRegressor().fit(X, y)
    print("predict:", model.predict(X[:4]))

    # 失败实验 1：拟合时 y 长度与 X 不一致 → ValueError
    try:
        MeanRegressor().fit([[1, 2], [3, 4]], [1, 2, 3])
    except ValueError as e:
        print("rejected bad length:", e)

    # 失败实验 2：predict 传入错误特征数 → ValueError
    try:
        model.predict(X[:, :2])
    except ValueError as e:
        print("rejected wrong n_features:", e)

    # 失败实验 3：未 fit 就 predict → ValueError('not fitted')
    try:
        MeanRegressor().predict(X)
    except ValueError as e:
        print("rejected not-fitted:", e)

    # 失败实验 4：输入含 NaN → ValueError
    X_nan = X.copy(); X_nan[0, 0] = np.nan
    try:
        MeanRegressor().fit(X_nan, y)
    except ValueError as e:
        print("rejected NaN:", e)

    assert np.allclose(model.predict(X[:4]), model.constant_)
    print("[PASS] s03: 数据契约统一 —— 非法输入产生明确错误，而非底层 NumPy 模糊异常")


if __name__ == "__main__":
    main()