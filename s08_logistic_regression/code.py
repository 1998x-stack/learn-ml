"""s08 Logistic Regression——从得分到概率再到类别 (v0.2.6-7).

Run:    python s08_logistic_regression/code.py

s06/s07 solved *regression*: predict a continuous value. This chapter turns to
**classification**: predict which of two categories a sample belongs to — but
the prediction is done *probabilistically*. The linear score
    z = X @ coef + intercept_
is fed into a **sigmoid** to become a probability p in (0, 1); the most
probable class is then read off via classes_ = argmax([1-p, p]).

Steps built from scratch (no imports beyond NumPy):
    1) a numerically-stable sigmoid   s(z) = 1 / (1 + exp(-z))
    2) the binary cross-entropy loss  -mean(y log p + (1-y) log(1-p))
    3) a small gradient-descent loop: w -= lr * Xa^T (p - y) / n
    4) decision_function / predict_proba / predict wrapped around score + sigmoid

The gradient of cross-entropy wrt the linear score is exactly (p - y), which is
why the update looks like a single line. That is the whole trick this chapter.
"""
# --- v0.2.6: sigmoid + 二元交叉熵 (binary cross-entropy) + 一个 GD 更新循环 ---
# --- v0.2.7: decision_function / predict_proba / predict + classes_ 标签编码 ---
from __future__ import annotations

import numpy as np


def sigmoid(z):
    """数值稳定的 sigmoid：z>=0 用 1/(1+exp(-z))，否则用 exp(z)/(1+exp(z))。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def binary_cross_entropy(y, p):
    """平均二元交叉熵为 -mean(y log p + (1-y) log(1-p))；p 先夹到非 0/1。"""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def separable_data(seed=7):
    """两块相距很远的 2D 高斯 blob —— 可分离，预测概率应接近 0 或 1。"""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-4.0, 0.6, (60, 2)), rng.normal(4.0, 0.6, (60, 2))])
    y = np.concatenate([np.repeat(0, 60), np.repeat(1, 60)])
    return X, y


def main() -> int:
    X, y = separable_data()

    # 标签编码：把 {a, b} 规整成 {0, 1}，class 1 = 排序后的“最大”类
    classes = np.unique(y)
    print(f"classes_ (标签编码) = {classes}")
    y01 = (y == classes[1]).astype(float)

    # 增广：X 前拼一列 1，让 intercept 成为 w 的第一个分量
    Xa = np.hstack([np.ones((X.shape[0], 1)), X])
    n = Xa.shape[0]
    rng = np.random.default_rng(0)
    w = rng.standard_normal(Xa.shape[1]) * 1e-3
    lr, max_iter = 0.5, 2000

    print("\n=== v0.2.6: sigmoid + 交叉熵 + GD，loss 应一路走低 ===")
    loss_history = [binary_cross_entropy(y01, sigmoid(Xa @ w))]
    for _ in range(max_iter):
        p = sigmoid(Xa @ w)
        grad = (1.0 / n) * (Xa.T @ (p - y01))   # 逻辑回归的梯度 = (p - y) 平均
        w = w - lr * grad
        loss_history.append(binary_cross_entropy(y01, sigmoid(Xa @ w)))
    n_dec = int(np.sum(np.diff(loss_history) < 0))
    print(f"loss {loss_history[0]:.4f} -> {loss_history[-1]:.4f} ({n_dec} 步下降)")

    # ——— v0.2.7 的三件套：把线性得分“翻译”成类别 ——
    intercept, coef = w[0], w[1:]

    def decision_function(X):
        return X @ coef + intercept

    def predict_proba(X):
        p = sigmoid(decision_function(X))
        return np.column_stack([1.0 - p, p])   # [P(class_0), P(class_1)]

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    print("\n=== v0.2.7: decision_function -> predict_proba -> predict ===")
    z = decision_function(X)
    print(f"decision_function 形状 {z.shape}，得分区间 [{z.min():.2f}, {z.max():.2f}]")

    proba = predict_proba(X)
    modal = proba.max(axis=1)
    print(f"predict_proba 行和≈1: {np.allclose(proba.sum(axis=1), 1.0)}，"
          f"多数类概率中位数 {np.median(modal):.3f}（可分离 → 接近 0/1）")

    acc = np.mean(predict(X) == y)
    print(f"若干可分离样本对 (P(class_0), P(class_1))：")
    for i in [0, 3, 60, 63]:
        print(f"  x={X[i].round(2)}  y={int(y[i])}  proba={proba[i].round(4)}  "
              f"predict={predict(X[i:i + 1])[0]}")

    # 教学自检：loss 收敛、决策形状、概率合法、准确率逼近完美
    assert loss_history[-1] < loss_history[0], "loss 应比起点更低"
    assert np.allclose(proba.sum(axis=1), 1.0), "概率行和必为 1"
    assert np.median(modal) > 0.99, "强分离样本：多数类概率应接近 1"
    assert acc > 0.99, "强分离数据应几乎全部判对"

    print(f"\n[PASS] s08: sigmoid + 交叉熵，得分 → 概率（predict_proba）→ 类别（predict），"
          f"训练准确率 {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())