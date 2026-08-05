"""s07 Optimization — 优化器负责寻找参数，模型负责定义目标 (v0.2.4-5).

Run:    python s07_optimization/code.py

s06 solved linear regression with a *closed form* (pinv) in one shot. When
X^T X is huge, slow to invert, or the model is non-linear, the closed form
disappears. This chapter introduces the other great family of optimizers:
**gradient descent (GD) and stochastic gradient descent (SGD)** — walk down
the loss surface one step at a time:

    w <- w - learning_rate * grad

We implement three things from scratch (no imports beyond NumPy):

    1) plain full-batch GD on MSE
    2) a learning-rate schedule: lr /= (1 + 0.01 * iter)  (decay over time)
    3) a tiny SGD that updates on a single shuffled sample per step

The core message: the *optimizer* finds params, the *model* defines the goal
(the loss function). The same loss can be minimized by GD or SGD or a closed
form — they are interchangeable under the same objective.
"""
# --- v0.2.4: GD 迭代优化——用梯度走下山，替代闭式解 pinv ---
# --- v0.2.5: 学习率调度 lr/(1+0.01*iter) + 迷你 SGD 单样本步进 ---
from __future__ import annotations

import numpy as np


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else float("nan")


def sgd_data(rng):
    X = np.linspace(-2, 2, 40)[:, None]
    y = 3.0 * X[:, 0] + 2.0 + 0.2 * rng.standard_normal(40)
    return X, y


def full_batch_gd(X, y, lr=0.05, max_iter=200, decay=False):
    """闭式解退位了——用梯度一步步走。返回 (w, loss_history)。"""
    n = len(y)
    Xa = np.hstack([np.ones((n, 1)), X])      # [intercept, coef...]
    w = np.zeros(Xa.shape[1])
    loss_history = [mse(y, Xa @ w)]
    for step in range(1, max_iter + 1):
        grad = (2.0 / n) * Xa.T @ (Xa @ w - y)   # 全量梯度
        lr_eff = lr / (1 + 0.01 * step) if decay else lr   # 学习率调度
        w = w - lr_eff * grad
        loss_history.append(mse(y, Xa @ w))
    return w, np.array(loss_history)


def mini_sgd(X, y, lr=0.05, epochs=50, seed=0):
    """迷你 SGD：每 epoch 洗牌，单样本梯度更新。返回 (w, loss_history)。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    Xa = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(Xa.shape[1])
    loss_history = [mse(y, Xa @ w)]
    for _ in range(epochs):
        for i in rng.permutation(n):
            xi = Xa[i]
            w = w - lr * 2.0 * xi * (xi @ w - y[i])   # 单样本梯度
        loss_history.append(mse(y, Xa @ w))
    return w, np.array(loss_history)


def show(tag, w, X, y, loss_hist):
    """打印收敛情况 + 真实参数 + R²。"""
    coef, intercept = w[1:], w[0]
    y_pred = X @ coef + intercept
    n_dec = int(np.sum(np.diff(loss_hist) < 0))
    print(f"[{tag}] coef_={coef.round(3)} intercept_={intercept:.3f} "
          f"(真值 3.0/2.0)  loss {loss_hist[0]:.3f}->{loss_hist[-1]:.4f} "
          f"({n_dec} 步下降)  R²={r2_score(y, y_pred):.3f}")


def main() -> int:
    rng = np.random.default_rng(42)
    X, y = sgd_data(rng)

    print("=== 1) 全批梯度下降（无调度）===")
    w_gd, h_gd = full_batch_gd(X, y)
    show("GD     ", w_gd, X, y, h_gd)

    print("\n=== 2) 学习率调度 lr/(1+0.01*iter)：大步到近旁，小步精修 ===")
    w_dc, h_dc = full_batch_gd(X, y, decay=True)
    show("GD+decay", w_dc, X, y, h_dc)

    print("\n=== 3) 迷你 SGD：单样本步进，噪声大但能处理超大语料 ===")
    w_sgd, h_sgd = mini_sgd(X, y, epochs=50)
    show("SGD    ", w_sgd, X, y, h_sgd)

    # 直观对比：三条曲线整体都应向下（SGD 有逐 epoch 噪声，允许小幅抖动）
    for tag, h in [("GD", h_gd), ("GD+decay", h_dc)]:
        assert np.all(np.diff(h) <= 1e-9), f"{tag} 的 loss 应单调不增"
    assert h_sgd[-1] < h_sgd[0], "SGD loss 应比起点更好"
    for tag, h in [("GD", h_gd), ("GD+decay", h_dc), ("SGD", h_sgd)]:
        assert h[-1] < h[0], f"{tag} 应比起点更好"

    # 三项最终都逼近真值 coef=3, intercept=2（教学验证，够近即可）
    assert np.allclose(w_gd[1], 3.0, atol=0.3) and np.isclose(w_gd[0], 2.0, atol=0.3)
    assert np.allclose(w_dc[1], 3.0, atol=0.3) and np.isclose(w_dc[0], 2.0, atol=0.3)
    assert np.allclose(w_sgd[1], 3.0, atol=0.9) and np.isclose(w_sgd[0], 2.0, atol=0.9)

    print("\n[PASS] s07: GD 全量步进、学习率调度、迷你 SGD——同一组参数三种找法")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
