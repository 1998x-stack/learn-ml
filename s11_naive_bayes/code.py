"""s11 Naive Bayes——从距离判断转向概率推断：先验 × 似然，全程对数空间 (v0.3.3).

Run:    python s11_naive_bayes/code.py

s10 的 KNN 靠「距离」判断：算 query 到每个邻居有多远、让最近的 k 个投票。
但距离这种实值比较，在类别多、维数高时会变贵且不稳定。s11 换成**概率推断**：
不再问"谁离得最近"，而是问"每种类别有多大的概率生成了这个样本"。

核心是贝叶斯公式：

    P(y | x)  ∝  P(y) × ∏_d P(x_d | y)

P(y) 是先验（一类样本占多少比例），P(x_d|y) 是"在类别 y 下、第 d 个特征取到
x_d 的似然"。"朴素"（naive）说的是我们**一带而过地假设给定类别后各特征相互独立**
，使联合似然能拆成每个特征各自的概率连乘——这个强假设换来模型简单、快速、稳定。

本文件「零导入 src」地手搓一套对数空间的高斯朴素贝叶斯：
    1) gaussian_log_pdf：单特征高斯对数概率密度
    2) 一个瘦 GaussianNB：fit 学每类每特征的均值 θ、方差 σ + 平滑
    3) 全程对数空间：logP(y|x) = log P(y) + Σ log p(x_d|y)，避免连乘下溢
    4) 上一小二维可分离数据集上验证：predict 还原类别、predict_proba 行和为 1
"""
# --- v0.3.3: log 空间 GaussianNB —— 强假设换来简单、快速、稳定 ---
from __future__ import annotations

import numpy as np

EPS = 1e-9  # 方差平滑：防止零方差导致除零


# --------------------------------------------------------------------------
# 1) 单个特征的高斯对数概率密度（log 空间是本章的灵魂）
# --------------------------------------------------------------------------
def gaussian_log_pdf(x, mu, var):
    """在 log 空间算 `N(x; mu, sigma^2)`，避免浓度数值爆炸/下溢。

    直接算 `1/sqrt(2 pi sigma^2) * exp(-(x-mu)^2/(2 sigma^2))` 会在 sigma 很小时
    溢出；取对数后变成一条加减乘除，稳得多。对数概率 = 各特征对数概率之和。
    """
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mu) ** 2 / np.maximum(var, EPS)


# --------------------------------------------------------------------------
# 2) 瘦 GaussianNB：fit 学每类别的均值/方差；predict 在对数空间做后验
# --------------------------------------------------------------------------
class GaussianNB:
    """对数空间高斯朴素贝叶斯分类器。

    fit：对每个类别 k，算出每特征 d 的均值 theta_[d,k] 与方差 sigma_[d,k]，
    并记录 class_log_prior_ = log(每类占比)。predict：对每个样本把"每特征的高斯
    对数概率"加起来，再叠加 log 先验，argmax 选后验最大的那类。
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        self.theta_ = []      # 每类别、每特征的均值
        self.sigma_ = []      # 每类别、每特征的方差（含平滑）
        self.class_log_prior_ = []
        for c in self.classes_:
            Xc = X[y == c]
            self.theta_.append(Xc.mean(axis=0))
            self.sigma_.append(Xc.var(axis=0) + self.var_smoothing)
            self.class_log_prior_.append(np.log(Xc.shape[0] / n))
        self.theta_ = np.array(self.theta_)      # (K, D)
        self.sigma_ = np.array(self.sigma_)      # (K, D)
        self.class_log_prior_ = np.array(self.class_log_prior_)  # (K,)
        return self

    def joint_log_likelihood(self, X):
        """返回 (n, K)：每行是 log P(y) + Σ_d log N(x_d; theta, sigma)。"""
        X = np.asarray(X, dtype=float)
        jll = np.empty((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            jll[:, k] = self.class_log_prior_[k]
            for d in range(self.n_features_in_):
                # 每个特征独立 → 对数概率直接相加（而不是密度相乘）
                jll[:, k] += gaussian_log_pdf(X[:, d], self.theta_[k, d], self.sigma_[k, d])
        return jll

    def predict_proba(self, X):
        # 对 joint log 做行内稳定 softmax：减每行最大值再求 exp，归一化
        jll = self.joint_log_likelihood(X)
        exp = np.exp(jll - jll.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(np.asarray(X, dtype=float)), axis=1)]


# --------------------------------------------------------------------------
# 3) 实验：两个靠得很近的二维高斯簇（"iris 近距离特征" 场景）+ 手写验证
# --------------------------------------------------------------------------
def make_close_blobs(seed=0):
    """两个中心只差一点的两个 2 维簇——正是后期特征"相似"、靠概率定类的场景。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 1.0], [2.2, 2.2]])  # 中心很近，只有 Bayesian 能分
    per = 60
    X = np.vstack([rng.normal(c, 0.4, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(0, per), np.repeat(1, per)])
    return X, y


def main() -> None:
    X, y = make_close_blobs()
    print("=== 高斯对数空间朴素贝叶斯 ===")
    nb = GaussianNB().fit(X, y)
    print("训练数据形状          =", X.shape)
    print("学到的每类均值 theta_ =")
    print("  类别 0:", np.round(nb.theta_[0], 2))
    print("  类别 1:", np.round(nb.theta_[1], 2))
    print("类别对数先验        =", np.round(nb.class_log_prior_, 3))

    pred = nb.predict(X)
    acc = float(np.mean(pred == y))
    P = nb.predict_proba(X)
    print("\n训练集 accuracy      =", round(acc, 4))
    print("predict_proba 行和   =", np.round(P.sum(axis=1)[:3], 6), "（应为 1）")
    proba = np.round(P[:3], 4)
    print("proba 首 3 行        =\n", proba)
    print("首 3 个预测标签      =", pred[:3])

    assert acc > 0.9
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(P > 0.0) and np.all(P < 1.0)
    print("\n[PASS] s11: 对数空间高斯朴素贝叶斯成立——用「先验 × 似然」的贝叶斯推断还原近距离二维簇")


if __name__ == "__main__":
    main()
