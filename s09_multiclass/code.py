"""s09 One-vs-Rest 多分类——不修改二分类器，用包装器扩展能力 (v0.2.8).

Run:    python s09_multiclass/code.py

s08 的 LogisticRegression 只会回答「是 / 不是」两个类。s09 要解决三流问题：
现在有 {triangle, square, circle} 三类，怎么办？最优雅的答案不是重写一个
softmax 模型，而是**复用**已经写好的二分类器——把多分类拆成 k 个二分类：

    OvR：对每个类别 c，训练一个「是不是 c」的二元分类器（其余当反例）。

本文件「零导入 src」地手搓这种包装器：
    1) sigmoid + 二元交叉熵（若从 s08 复习，就直接沿用那份逻辑）
    2) 一个手写的 binary 二分类 GD 循环（每类各跑一遍）
    3) LabelEncoder：把任意标签压缩成 0..k-1 的密集整数
    4) OneVsRest：k 个模型各吐「是第 c 类」的概率，argmax 定胜负

核心思想——**元估计器（meta-estimator）**：包装器不改基础模型，只把它
组合进一个更大的算法。基础二分类器不知道自己被用于多分类。
"""
# --- v0.2.8: OvR 多分类——每个类别克隆一个二分类器，argmax 决策 ---
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
    """平均二元交叉熵 -mean(y log p + (1-y) log(1-p))；p 先夹到非 0/1。"""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class BinaryLogistic:
    """手写二元逻辑回归（复刻 s08，仅此一份，供 OvR 循环 k 次调用）。

    fit 之后暴露 coef_ / intercept_ / predict_proba(X)：
    predict_proba 返回 (n,2) 的 [1-p, p]，p 是「正类」概率。
    """

    def __init__(self, learning_rate=0.5, max_iter=2000, random_state=0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y_bin):
        """y_bin 已是 0/1；用全批 GD 最小化二元交叉熵。"""
        Xa = np.hstack([np.ones((X.shape[0], 1)), X])          # 增广：X 前拼 1
        n = Xa.shape[0]
        rng = np.random.default_rng(self.random_state)
        w = rng.standard_normal(Xa.shape[1]) * 1e-3
        self.loss_history_ = [binary_cross_entropy(y_bin, sigmoid(Xa @ w))]
        for _ in range(self.max_iter):
            p = sigmoid(Xa @ w)
            grad = (1.0 / n) * (Xa.T @ (p - y_bin))            # 逻辑回归梯度
            w = w - self.learning_rate * grad
            self.loss_history_.append(binary_cross_entropy(y_bin, sigmoid(Xa @ w)))
        self.intercept_ = float(w[0])
        self.coef_ = np.asarray(w[1:], dtype=float)
        return self

    def predict_proba(self, X):
        """(n,2) 概率表；col=1 是「正类」概率。"""
        z = X @ self.coef_ + self.intercept_
        p = sigmoid(z)
        return np.column_stack([1.0 - p, p])


class LabelEncoder:
    """把任意标签压成 0..k-1 的密集整数，fit 时记住 classes_。"""

    def fit(self, y):
        self.classes_ = np.unique(y)
        self._idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self._idx[v] for v in y])

    def inverse_transform(self, encoded):
        return self.classes_[np.asarray(encoded, dtype=int)]


class OneVsRestClassifier:
    """元估计器：对每个类克隆一个 BinaryLogistic 拟合「是不是 c」。"""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimators_ = []                                  # k 个克隆，互不干扰
        for c in self.classes_:
            y_bin = (y == c).astype(float)                    # 一对一 vs 其余
            est = self._clone(self.estimator)                 # 每个类一份新模型
            est.fit(X, y_bin)
            self.estimators_.append(est)
        return self

    @staticmethod
    def _clone(estimator):
        # 用相同超参数重建一个未训练实例（等价于框架的 clone）
        return estimator.__class__(
            learning_rate=estimator.learning_rate,
            max_iter=estimator.max_iter,
            random_state=estimator.random_state,
        )

    def _scores(self, X):
        # 把每个克隆的「正类概率」聚成 (n, k) 的得分矩阵
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])

    def predict(self, X):
        scores = self._scores(X)
        best = np.argmax(scores, axis=1)                      # 同一类 → 最大得分
        return self.classes_[best]

    def predict_proba(self, X):
        scores = self._scores(X)
        return scores / scores.sum(axis=1, keepdims=True)     # 归一化成概率


def three_class_data(seed=3):
    """三个相距很远的 2D 高斯 blob：可分离，OvR 应几乎全对。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[-4.0, -4.0], [4.0, -4.0], [0.0, 4.0]])
    per = 60
    X = np.vstack([rng.normal(c, 0.5, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(i, per) for i in range(3)])
    return X, y


def main() -> int:
    X, y = three_class_data()
    n, k = X.shape[0], len(np.unique(y))

    # ——— LabelEncoder：任意标签 → 0..k-1 的密集整数 —
    enc = LabelEncoder().fit(y)
    print(f"classes_ (排序后)   = {enc.classes_}")
    print(f"transform(前 6 个)  = {enc.transform(y[:6])}")
    print(f"inverse 回原标签    = {enc.inverse_transform(enc.transform(y[:6]))}")

    # ——— OvR：k 个独立的二元逻辑回归，各管一个「是不是 c」 —
    clf = OneVsRestClassifier(BinaryLogistic())
    clf.fit(X, y)
    print("\n=== 元估计器：每个克隆拟合 (X, y=='c') 的二元问题 ===")
    print(f"estimators_ 数量    = {len(clf.estimators_)}（= 类别数 {k}）")

    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    acc = np.mean(pred == y)
    print("\n=== argmax 决策 ===")
    print(f"得分矩阵形状        = {proba.shape}（{n} 个样本 × {k} 类）")
    print(f"行和 ≈ 1            = {np.allclose(proba.sum(axis=1), 1.0)}")

    print("\n若干样本的真实类 vs 预测类 vs 概率：")
    for i in [0, 60, 120, 5, 65, 125]:
        print(f"  y={int(y[i])}  predict={int(pred[i])}  "
              f"proba={proba[i].round(3)}")

    # 教学自检：OvR 三个模型，训练准确率高，概率合法
    assert len(clf.estimators_) == 3, "激列表应恰为 3 个"
    assert np.allclose(proba.sum(axis=1), 1.0), "概率行和必为 1"
    assert acc > 0.8, "可分离三丘数据上 OvR 应高准确率"
    assert np.all(enc.inverse_transform(enc.transform(y)) == y), "标注编码回环必自洽"

    print(f"\n[PASS] s09: 不改一只用 k 个二分类器的包装，得到多分类，"
          f"OvR 训练准确率 {acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
