"""s06 Linear Regression — 数据不再只决定一个常数，而是决定一组参数 (v0.2.3).

Run:    python s06_linear_regression/code.py

s01–s05 always predicted a constant (a class, a fill value, a scaled state).
s06 is the FIRST real learner: closed-form least-squares linear regression.
We solve Xw = y with two equivalent recipes:

    1) normal equation   w = (X^T X)^{-1} X^T y   (needs X^T X invertible)
    2) pseudoinverse     w = np.linalg.pinv(X) @ y (works even when X^T X is
                            singular / X is tall / redundant columns)

The key difference: pinv handles rank-deficient X (e.g. more features than
samples, or duplicate columns); the normal equation breaks there. sklearn's
LinearRegression uses pinv. We follow it.

This file is self-contained: it re-implements the learner inline (teaching
track) instead of importing the framework package.
"""
# --- v0.2.3: 首个真正的学习器 LinearRegression/Ridge：闭式解 Xw=y（np.linalg.pinv）---
from __future__ import annotations

import numpy as np


class LinearRegression:
    """Closed-form least-squares linear regression (self-contained, pinv).

    fit_intercept: prepends a column of ones to X so the bias is solved as
    part of the augmented weight vector w_aug = [intercept, coef...].
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            Xa = np.hstack([np.ones((X.shape[0], 1)), X])
            w = np.linalg.pinv(Xa) @ y      # least-squares / pseudoinverse
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            w = np.linalg.pinv(X) @ y
            self.coef_ = w
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def r2_score(y_true, y_pred):
    """Coefficient of determination, guarded for constant target."""
    ss_res = np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else float("nan")


def fit_1d_closed_form(x, y):
    """Single-feature closed-form slope/intercept via raw sums.

    Useful before generalizing: for one feature there is a hand-computable
    formula (no matrices at all). We then compare it to the pinv result.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_bar, y_bar = x.mean(), y.mean()
    slope = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    intercept = y_bar - slope * x_bar
    return slope, intercept


def main() -> int:
    rng = np.random.default_rng(42)

    print("=== 1) 单特征手算闭式解（先看简单的）===")
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])          # y = 1.0*x + 1.0，无噪声
    slope, intercept = fit_1d_closed_form(x, y)
    print(f"手算 slope={slope:.6f}, intercept={intercept:.6f}  (期望 1.0 / 1.0)")

    # 用 pinv 验证：加一列 1 再伪逆，得到同一个 (intercept, coef)
    X1 = x.reshape(-1, 1)
    lr1 = LinearRegression(fit_intercept=True).fit(X1, y)
    y_pred = lr1.predict(X1)
    print(f"pinv coef_={lr1.coef_.round(6)}, intercept_={lr1.intercept_:.6f}")
    print(f"R^2 = {r2_score(y, y_pred):.4f}")
    assert np.allclose(lr1.coef_, [1.0]) and np.isclose(lr1.intercept_, 1.0)

    print("\n=== 2) 多特征 + 噪声：伪逆（pinv）===")
    true_w = np.array([2.0, -1.0, 0.5])
    Xm = rng.standard_normal((500, 3))
    ym = Xm @ true_w + 3.0 + 0.1 * rng.standard_normal(500)
    lr_m = LinearRegression(fit_intercept=True).fit(Xm, ym)
    print(f"coef_  {lr_m.coef_.round(3)}   (真值 {true_w})")
    print(f"intercept_ {lr_m.intercept_:.3f}   (真值 3.0)")
    print(f"R^2 = {r2_score(ym, lr_m.predict(Xm)):.4f}")
    assert np.allclose(lr_m.coef_, true_w, atol=0.05)
    assert np.isclose(lr_m.intercept_, 3.0, atol=0.05)

    print("\n=== 3) pinv 在 X^T X 奇异时仍能解（normal-form 会炸）===")
    x2 = np.linspace(0, 1, 20)
    Xdup = np.column_stack([x2, x2])            # 两列完全相同 -> X^T X 奇异
    y2 = 2.0 * x2 + 1.0
    lr_dup = LinearRegression(fit_intercept=True).fit(Xdup, y2)
    print(f"  重复特征下 coef_={lr_dup.coef_.round(3)} R^2={r2_score(y2, lr_dup.predict(Xdup)):.4f}")
    # 若用 normal-form (X^T X)^{-1} 会因奇异矩阵 linrecur 报错；pinv 受到呵护
    assert np.isfinite(lr_dup.coef_).all()

    print("[PASS] s06: 线性回归闭式解——手算 slope、pinv 多特征、奇异列落于 rank-deficient 仍可解")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
