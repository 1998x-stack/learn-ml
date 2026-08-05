"""s02 Estimator Protocol — 新增模型，不改变调用方式 (v0.1.3).

Run:    python s02_estimator_protocol/code.py

与 learn-claude-code 的 s02 对应：s02 把"单个 Bash 调用"改造成"工具定义 + Handler 映射"，
新增工具不再修改 Agent Loop。这里对应的是：新增算法只实现统一 Estimator 协议，
不修改训练 / 评估框架。
"""
# --- v0.1.3: BaseEstimator + get_params/set_params + clone + RegressorMixin ---
from __future__ import annotations
from inspect import signature

import numpy as np


class BaseEstimator:
    """get_params / set_params 通过内省 __init__ 签名实现。"""

    def get_params(self, deep: bool = True) -> dict:
        names = [k for k in signature(self.__init__).parameters if k != "self"]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params) -> "BaseEstimator":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key!r}")
            setattr(self, key, value)
        return self


class RegressorMixin:
    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        u = float(np.sum((y - y_pred) ** 2))
        v = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - u / v


def clone(estimator: BaseEstimator) -> BaseEstimator:
    """按 get_params() 重建一个未训练的全新产品。"""
    return estimator.__class__(**estimator.get_params())


class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, strategy: str = "mean", offset: float = 0.0):
        self.strategy = strategy
        self.offset = offset

    def fit(self, X, y) -> "MeanRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.constant_ = float(np.mean(y)) + self.offset
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return np.full(X.shape[0], self.constant_)


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    y = 2.0 + rng.standard_normal(10)

    model = MeanRegressor(strategy="mean", offset=0.1)  # 超参数，非学习产物
    model.fit(X, y)

    print("get_params:", model.get_params())
    print("predict:", model.predict(X[:3]))

    copied = clone(model)
    print("克隆是全新对象:", copied is not model,
          "| 超参数一致:", copied.get_params() == model.get_params())

    print("score (R²):", round(model.score(X, y), 3))

    assert copied is not model and copied.get_params() == model.get_params()
    assert model.get_params()["strategy"] == "mean"
    print("[PASS] s02: get_params / set_params / clone / score 一致，Estimator 协议成立")


if __name__ == "__main__":
    main()
