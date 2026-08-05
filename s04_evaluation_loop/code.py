"""s04 Evaluation Loop — 训练、预测、评估形成闭环 (v0.1.5).

Run:    python s04_evaluation_loop/code.py
"""
# --- v0.1.5: train_test_split + metrics（MSE/MAE/R²/accuracy）形成首个评估闭环 ---
from __future__ import annotations

import numpy as np


def train_test_split(X, y, test_size=0.25, random_state=None):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_test = int(np.ceil(n * test_size))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def mean_squared_error(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


class MeanRegressor:
    def fit(self, X, y):
        self.constant_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.constant_)


class MostFrequentClassifier:
    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.most_frequent_ = classes[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.most_frequent_)


def main() -> None:
    rng = np.random.default_rng(42)

    # 回归数据
    X = rng.standard_normal((200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(200) * 0.1

    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = MeanRegressor().fit(X_tr, y_tr)
    pred = reg.predict(X_te)

    print("=== 回归 ===")
    print(f"MSE: {mean_squared_error(y_te, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_te, pred):.4f}")
    print(f"R² : {r2_score(y_te, pred):.4f}")
    # 回归断言：pred 与 y_te（回归目标）同源，指标可计算
    assert pred.shape == (40,)
    assert np.isfinite(mean_squared_error(y_te, pred))
    assert np.isfinite(mean_absolute_error(y_te, pred))
    assert np.isfinite(r2_score(y_te, pred))

    # 分类（基线分类器）
    Xc = rng.standard_normal((200, 3))
    yc = (Xc[:, 0] + Xc[:, 1] + rng.standard_normal(200) > 0).astype(int)
    Xc_tr, yc_tr, Xc_te, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
    clf = MostFrequentClassifier().fit(Xc_tr, yc_tr)
    yc_pred = clf.predict(Xc_te)
    acc = accuracy_score(yc_te, yc_pred)

    print("=== 分类 ===")
    print(f"accuracy: {acc:.4f}")

    # 分类断言：accuracy 在 [0,1]；分类标签与预测同源
    assert yc_pred.shape == (40,)
    assert yc_pred.shape[0] == yc_te.shape[0]
    assert 0.0 <= acc <= 1.0
    print("[PASS] s04: 独立测试集 + 评估闭环 —— 没有独立测试集就没有可信效果")


if __name__ == "__main__":
    main()
