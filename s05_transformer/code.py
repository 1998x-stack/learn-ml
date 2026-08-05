"""s05 Transformer — 学习数据状态，但不预测目标 (v0.2.1 → v0.2.2).

Run:    python s05_transformer/code.py

A Transformer (StandardScaler / MinMaxScaler / SimpleImputer) learns some
*state* of the data during fit (mean / std / min / max / NaN statistics) but
NEVER predicts a target. It hands that learned state to transform(), which
rewrites the input.

The anti-pattern to avoid:
    StandardScaler().fit_transform(X_test)
recomputes the test set's OWN mean/std and applies them — leaking the test
set's statistics into the transformation. transform() must always REUSE the
statistics learned from X_train.
"""
# --- v0.2.1: 框架轨道新增 preprocessing（StandardScaler/MinMaxScaler/SimpleImputer）---
# --- v0.2.2: 教学轨道 code.py 演示 fit/transform 如何复用训练统计，杜绝测试集统计泄露 ---
from __future__ import annotations

import numpy as np


class StandardScaler:
    """Center & scale each feature to zero mean / unit variance (self-contained)."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xt = X - self.mean_ if self.with_mean else X
        return Xt / self.scale_ if self.with_std else Xt

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale each feature to [0, 1]; guard zero-range features (self-contained)."""

    def __init__(self, with_min: float = 0.0, with_max: float = 1.0):
        self.with_min = with_min
        self.with_max = with_max

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        scale = X.max(axis=0) - self.min_
        self.scale_ = np.where(scale == 0, 1.0, scale)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        span = self.with_max - self.with_min
        return (X - self.min_) / self.scale_ * span + self.with_min

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class SimpleImputer:
    """Fill NaN with a per-column statistic (self-contained)."""

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        else:
            raise ValueError(f"Invalid strategy {self.strategy!r}")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        missing = np.isnan(X)
        X = X.copy()
        X[missing] = np.take(self.statistics_, np.where(missing)[1])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def main() -> None:
    rng = np.random.default_rng(42)

    # 造一个分布略有差异的训练/测试对
    X_train = rng.standard_normal((200, 3)) * 2.0 + 1.0
    X_test = rng.standard_normal((50, 3)) * 2.5 + 0.8

    print("=== 反模式：对测试集重复 fit_transform（数据泄露）===")
    # anti-pattern: recomputes test statistics from the test set itself
    leaked = StandardScaler().fit_transform(X_test)
    print(f"测试集被自己的统计量标准化 -> 测试集均值为 {leaked.mean(axis=0).round(3)}，"
          f"标准差 {leaked.std(axis=0).round(3)}，物理含义被抹除")
    # 泄露的后果：测试集的统计被"改写"，模型看不到它真实分布的形态

    print("\n=== 正确模式：fit 在训练集，transform 复用统计量 ===")
    sc = StandardScaler().fit(X_train)          # 只学一次：训练集的 mean_/scale_
    Xtr_scaled = sc.transform(X_train)          # 复用同一次统计
    Xte_scaled = sc.transform(X_test)           # 同一套统计应用到测试集
    print(f"训练集缩放后均值   {Xtr_scaled.mean(axis=0).round(4)}，std {Xtr_scaled.std(axis=0).round(4)}")
    print(f"测试集缩放后均值   {Xte_scaled.mean(axis=0).round(4)}，std {Xte_scaled.std(axis=0).round(4)}")
    # 测试集均值 ≈ (0.8 - 1.0)/2.0 = -0.1 附近，std ≈ 2.5/2.0 = 1.25 —— 保留真实形状

    mm = MinMaxScaler().fit(X_train)
    Xtr_mm = mm.transform(X_train)
    Xte_mm = mm.transform(X_test)
    print(f"MinMax 训练集范围 [{Xtr_mm.min(axis=0).round(4)}, {Xtr_mm.max(axis=0).round(4)}]")

    # 反模式断言：泄露的测试集均值 ≈ 0（统计被抹除）
    assert np.allclose(leaked.mean(axis=0), 0.0, atol=1e-12), "反模式应让测试集均值归零"
    # 正确模式断言：测试集统计量来自训练集，不会被抹平
    assert not np.allclose(Xte_scaled.mean(axis=0), 0.0, atol=1e-3), "正确模式保留测试集形状"

    print("\n=== SimpleImputer：用一列的非缺失统计填充缺失值 ===")
    X_nan = rng.standard_normal((100, 3))
    X_nan[::5, 0] = np.nan                       # 每 5 行挖掉一列的一个值
    imp = SimpleImputer(strategy="mean").fit(X_nan)
    X_filled = imp.transform(X_nan)
    assert not np.isnan(X_filled).any()
    print(f"缺失填充后：无 NaN，列均值 {X_filled[:, 0].mean(axis=0).round(4)} 近似原列均值 "
          f"{np.nanmean(X_nan[:, 0]).round(4)}")

    print("[PASS] s05: Transformer 学习数据状态但不预测目标；transform 复用训练统计而非重算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
