"""s12 decision tree——一次划分很弱，递归划分形成复杂边界 (v0.3.4-5).

Run:    python s12_decision_tree/code.py

s10（KNN）用「距离」、s11（NB）用「概率」判断一个样本。s12 换一种完全不同的
思路：**用一组递归的 if-else 规则**去切分特征空间。一颗决策树就是一个
「可解释的分层问句」：
    特征 x0 <= 2.5 ?  →  是 → 往左走（再问更细的问题）
                       否 → 往右走

单看一次划分（一个 if-else）很弱——一条直线切一个特征，能表达的边界很有限。
但**递归划分**：先在特征上切一刀，再在每边各自继续切，就能拼出非常复杂的
决策边界。这就是本章口号：**一次划分很弱，递归划分形成复杂边界。**

下面「零导入 src」地手搓一套递归二分决策树：
    1) gini 不纯度（v0.3.4）：1 - Σ p²，纯节点=0、「越平均越杂」越大
    2) 找最优划分：对每个特征排序、在相邻不同值的中点试刀，取加权不纯度最小
    3) 递归生长：选完根分裂，对左右递归长出子树；叶子存多数类 / 回归均值
    4) entropy（v0.3.5）：-Σ p log2 p；回归树用平方误差（方差）当不纯度
    5) feature_importances_：每特征贡献的整体加权不纯度下降，归一化求和≈1
"""
# --- v0.3.4: 递归二分（gini）——一次划分弱、递归划分强 ---
# --- v0.3.5: +entropy 准则 + 回归树（平方误差/方差） ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# 0) 树节点：叶子记 value；内部节点记 feature/threshold + 两个孩子
# --------------------------------------------------------------------------
@dataclass
class Node:
    value: Optional[object] = None          # 叶子：预测（多数类，或回归均值）
    feature: Optional[int] = None           # 内部：切哪个特征
    threshold: Optional[float] = None       # 内部：x[feature] <= threshold 走左
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    n_samples: float = 0.0
    impurity: float = 0.0


# --------------------------------------------------------------------------
# 1) 不纯度：gini、熵与方差（MSE）——「越纯越小，纯则=0」
# --------------------------------------------------------------------------
def gini(counts):
    """1 - Σ p_k²。某类占 100%（纯）→ 0；两类各 50% → 0.5。"""
    p = counts / counts.sum()
    return float(1.0 - (p * p).sum())


def entropy(counts):
    """信息熵 -Σ p log2 p：纯节点熵=0，越不确定熵越大。"""
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def variance(y, w):
    """加权方差：回归树的不纯度（叶子内平方误差的平均）。"""
    mean = (w * y).sum() / w.sum()
    return float((w * (y - mean) ** 2).sum() / w.sum())


def impurity_of(y, w, criterion):
    """把一个标签数组转成加权计数，再套 gini / 熵 / 方差。"""
    if criterion == "squared_error":
        return variance(y, w)
    counts = np.bincount(y.astype(int), weights=w)
    return gini(counts) if criterion == "gini" else entropy(counts)


# --------------------------------------------------------------------------
# 2) 最优划分：对每个特征排序、在相邻不同值的中点试刀，取加权不纯度最小
# --------------------------------------------------------------------------
def best_split(X, y, w, criterion):
    n, n_features = X.shape
    parent = impurity_of(y, w, criterion)
    best_score = parent
    best = (None, None, parent)
    for f in range(n_features):
        order = np.argsort(X[:, f], kind="mergesort")
        xs, ys, ws = X[order, f], y[order], w[order]
        for i in range(1, n):
            if xs[i] <= xs[i - 1]:
                continue                       # 相邻值相同 → 无有效中点
            threshold = (xs[i - 1] + xs[i]) / 2.0
            imp = (i / n) * impurity_of(ys[:i], ws[:i], criterion) + \
                  ((n - i) / n) * impurity_of(ys[i:], ws[i:], criterion)
            if imp < best_score:
                best_score = imp
                best = (f, threshold, imp)
    return best


# --------------------------------------------------------------------------
# 3) 递归生长：先当叶子，能切就切 → 左右递归；depth/规模上限保护递归深度
# --------------------------------------------------------------------------
def grow_tree(X, y, w, depth, criterion, max_depth, min_samples_split):
    node = Node(
        value=_leaf_value(y, w, criterion),
        n_samples=w.sum(),
        impurity=impurity_of(y, w, criterion),
    )
    if node.impurity == 0.0:
        return node                                   # 已经纯了
    if max_depth is not None and depth >= max_depth:
        return node                                   # 到深度上限
    if len(y) < min_samples_split or len(np.unique(y)) < 2:
        return node                                   # 样本太少 / 单类
    f, thr, _ = best_split(X, y, w, criterion)
    if f is None:
        return node                                   # 找不到更纯的划分
    left_mask = X[:, f] <= thr
    if left_mask.all() or not left_mask.any():
        return node                                   # 退化切分，切成空/全
    node.feature, node.threshold = f, thr
    node.left = grow_tree(X[left_mask], y[left_mask], w[left_mask],
                          depth + 1, criterion, max_depth, min_samples_split)
    node.right = grow_tree(X[~left_mask], y[~left_mask], w[~left_mask],
                           depth + 1, criterion, max_depth, min_samples_split)
    return node


def _leaf_value(y, w, criterion):
    """叶子值：分类=多数类；回归=加权均值。"""
    if criterion == "squared_error":
        return float((w * y).sum() / w.sum())
    return int(np.argmax(np.bincount(y.astype(int), weights=w)))


# --------------------------------------------------------------------------
# 4) 预测（沿树走到叶子）与特征重要性
# --------------------------------------------------------------------------
def predict_row(row, node):
    while node.left is not None and node.right is not None:
        node = node.left if row[node.feature] <= node.threshold else node.right
    return node.value


def feature_importances(root, n_features):
    """每个特征贡献的总加权不纯度下降，归一化求和≈1。"""
    gains = np.zeros(n_features)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.left is None:
            continue
        split_imp = (node.left.n_samples / node.n_samples * node.left.impurity
                     + node.right.n_samples / node.n_samples * node.right.impurity)
        gains[node.feature] += (node.n_samples / root.n_samples) * (
            node.impurity - split_imp)
        stack += [node.left, node.right]
    return gains / gains.sum()


def print_tree(node, indent=0):
    pad = "  " * indent
    if node.left is None:
        print(f"{pad}[叶子: 值={node.value} n={node.n_samples:.0f}]")
    else:
        print(f"{pad}[x{node.feature} <= {node.threshold:.2f} "
              f"imp={node.impurity:.3f} n={node.n_samples:.0f}]")
        print_tree(node.left, indent + 1)
        print_tree(node.right, indent + 1)


# --------------------------------------------------------------------------
# 5) 实验：小二维可分离集（一条斜线就能分开）+ 一元回归
# --------------------------------------------------------------------------
def make_tiny_separable(seed=0):
    """两类各一个三角分布，几乎线性可分；训练数据少、树浅、绝不爆深度。"""
    rng = np.random.default_rng(seed)
    c0 = np.array([[-1.5, -1.0], [0.0, 0.0], [-2.0, 0.5]])     # 左下，类 0
    c1 = np.array([[1.5, 1.0], [0.0, 0.0], [2.0, -0.5]])       # 右上，类 1
    X = np.vstack([c0 + rng.normal(0, 0.15, c0.shape),
                   c1 + rng.normal(0, 0.15, c1.shape)])
    y = np.concatenate([np.zeros(len(c0)), np.ones(len(c1))])
    return X, y


def main() -> None:
    X, y = make_tiny_separable()
    w = np.ones(len(y))

    root = grow_tree(X, y, w, 0, "gini", max_depth=None, min_samples_split=2)
    yhat = np.array([predict_row(r, root) for r in X])
    acc = float(np.mean(yhat == y))
    imp = feature_importances(root, X.shape[1])

    print("=== 递归二分决策树 (gini, v0.3.4) ===")
    print("数据形状         =", X.shape)
    print("生长出的树：")
    print_tree(root)
    print("训练集 accuracy =", round(acc, 4))
    print("特征重要性      =", np.round(imp, 4), " 求和=%.4f" % imp.sum())

    root_e = grow_tree(X, y, w, 0, "entropy", max_depth=2, min_samples_split=2)
    acc_e = float(np.mean([predict_row(r, root_e) for r in X] == y))
    print("\n=== 信息熵 criterion=entropy (v0.3.5) ===")
    print("entropy 树 train acc =", round(acc_e, 4))

    Xr = np.linspace(-2, 2, 8).reshape(-1, 1)
    yr = 2.0 * Xr[:, 0]                                   # 线性目标 y=2x0
    reg = grow_tree(Xr, yr, np.ones(len(yr)), 0, "squared_error",
                    max_depth=2, min_samples_split=2)
    yreg = np.array([predict_row(r, reg) for r in Xr])
    print("\n=== 回归树 DecisionTreeRegressor (v0.3.5) ===")
    print("目标 y = 2*x0；叶子均值预测 =", np.round(yreg, 3))
    print("所有预测有限        =", bool(np.isfinite(yreg).all()))

    assert acc > 0.98, "separable tiny data should be (near-)perfect"
    assert abs(imp.sum() - 1.0) < 1e-6, "feature_importances_ 求和应≈1"
    print("\n[PASS] s12: 递归二分（gini/entropy）长成可解释树——一刀弱、递归强")


if __name__ == "__main__":
    main()
