"""s10 Lazy Learning——pairwise 距离 + KNN：模型也可以不学习显式参数 (v0.3.1-2).

Run:    python s10_neighbors/code.py

s06 的线性回归学出一份「权重」，s08 的逻辑回归学出一份「coef + intercept」——
它们都属于 **急切学习（eager）**：训练期花费大量算力，predict 却只是扫一遍公式。
s10 反着来：**懒学习（lazy）**——fit 阶段什么都不学，只把训练数据原样存下来；
predict 阶段才开始真正干活：对每个待预测点，去训练集里搜索最邻近的 k 个点，
让这些「邻居」投票 / 取平均来定答案。

    fit：保存训练数据              →  训练很轻
    predict：搜索最近邻再聚合     →  推理很重

本文件「零导入 src」地手搓一套：
    1) euclidean/ 实现 pairwise 欧氏距离矩阵（先 Python 双循环，再优化成 NumPy 广播）
    2) iterate 一个瘦 KNN：fit 存 X，predict 对每个点遍历找 k 近邻、多数表决
    3) kNN Regressor：predict 取 k 近邻 y 的均值
    4) 在二维可分离合成数据上验证，predict 精确还原类别
"""
# --- v0.3.1: pairwise（Python 双循环 → NumPy 广播）+ 手写 KNN 分类器 ---
# --- v0.3.2: 同样的 KNN 复用去当回归器 —— fit 只存数据，predict 聚合邻居 ---
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 1) pairwise：学会把「样本对的平方距离」写成一条广播表达式
# --------------------------------------------------------------------------
def pairwise_naive(X, Y):
    """Python 双循环：每对样本手算一次欧氏距离（教学，先看懵在哪）。"""
    n, ny = X.shape[0], Y.shape[0]
    D = np.empty((n, ny))
    for i in range(n):
        for j in range(ny):
            D[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return D


def euclidean_distances(X, Y=None):
    """NumPy 广播：一次算完 (n_x, n_y, d) 的差，再沿特征轴开方求和。

    ``X[:, None, :]`` 是 (n_x, 1, d)，``Y[None, :, :]`` 是 (1, n_y, d)，
    点乘广播成 (n_x, n_y, d)；``axis=-1`` 把特征维约掉。Y=None 表示「跟
    自己比」，此时对角恰好 = 0。
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _check_pairwise_consistent():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    naive = pairwise_naive(X, X)
    brute = euclidean_distances(X, X)
    return bool(np.allclose(naive, brute))


# --------------------------------------------------------------------------
# 2) 懒学习 KNN 分类器：fit 存 X，predict 才搜索邻居
# --------------------------------------------------------------------------
class KNNClassifier:
    """瘦 KNN（懒学习）：fit 只存数据；predict 对每个点找 k 近邻、多数投票。

    与 s06/s08 相反——s06 的 LinearRegression 拟合出权重 w，SGD 那里的“训练”
    地在 iter 循环里一点点磨参数；这里 fit 一行代码都不优化任何东西，真正的
    工作被「推」到了 predict。这正是「训练很轻、推理很重」。
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)   # 这就是“学到的模型”本体
        self.y_train = np.asarray(y)
        return self

    def predict_point(self, x):
        """单个待测点：先算它到所有训练点的距离，再取前 k 个的类投票。"""
        # (1, d) 广播减 (n, d) -> (n, d) —— per-sample 的 mini pairwise
        d = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
        k_ix = np.argsort(d)[: self.k]
        k_labels = self.y_train[k_ix]
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # 教学版：对每个待测点循环（predict 才忙），对应内层 "per point" 的 k 搜索
        return np.array([self.predict_point(x) for x in X])


class KNNRegressor(KNNClassifier):
    """把「邻居多数投票」换成「邻居均值」就得到回归器——同一个懒框架。

    predict 不再 argmax 一个类别，而是对 k 个邻居的连续 y 求平均。KNN 的
    分类/回归共享同一套距离 + 选邻居逻辑，只看最终怎么「聚合」邻居。"""

    def predict_point(self, X):
        d = np.sqrt(((self.X_train - X) ** 2).sum(axis=1))
        k_ix = np.argsort(d)[:self.k]
        return float(self.y_train[k_ix].mean())


# --------------------------------------------------------------------------
# 3) 实验：二维可分离合成数据 + 手写验证
# --------------------------------------------------------------------------
def make_two_clusters(seed=0):
    """两个相距很远的二维高斯簇，KNN 能几乎 100% 还原标签。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[-3.0, -3.0], [3.0, 3.0]])
    per = 50
    X = np.vstack([rng.normal(c, 0.7, (per, 2)) for c in centers])
    y = np.concatenate([np.repeat(0, per), np.repeat(1, per)])
    return X, y


def main() -> None:
    print("=== pairwise：Python 双循环 vs NumPy 广播 ===")
    print("两实现一致?", _check_pairwise_consistent())
    X_pair = np.array([[0.0, 0.0], [3.0, 4.0]])
    print("euclidean_distances([[0,0],[3,4]]) =\n", euclidean_distances(X_pair))

    X, y = make_two_clusters()
    print("\n=== 懒学习 KNN 分类器 ===")
    knn = KNNClassifier(k=5).fit(X, y)
    pred = knn.predict(X)
    acc = float(np.mean(pred == y))
    print("训练数据形状          =", X.shape, "（fit 阶段只是原样保存）")
    print("可分离二维数据 accuracy =", round(acc, 3))
    print("前 6 个预测标签      =", pred[:6])
    assert acc > 0.95

    print("\n=== 同一个懒框架：KNN 回归器（邻居均值）===")
    coef = np.array([2.0, -1.0])
    y_reg = X @ coef + 0.1
    rng = np.random.default_rng(1)
    y_reg = y_reg + rng.normal(0.0, 0.05, X.shape[0])
    knnr = KNNRegressor(k=5).fit(X, y_reg)
    pred_reg = knnr.predict(X)
    print("回归预测形状          =", pred_reg.shape)
    print("回归预测全有限        =", bool(np.isfinite(pred_reg).all()))
    mae = float(np.mean(np.abs(pred_reg - y_reg)))
    print("回归 MAE              =", round(mae, 3), "（邻居均值逼近真函数）")

    # 教学自检：KNN 分类在可分离聚类上应还原标签
    assert np.allclose(euclidean_distances(X, X), euclidean_distances(X))
    assert acc > 0.95 and pred.shape == (X.shape[0],)
    print("\n[PASS] s10: 懒学习成立——fit 只存数据、predict 才找邻居；KNN 还原二维可分离聚类")


if __name__ == "__main__":
    main()
