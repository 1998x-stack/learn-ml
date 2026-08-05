"""s01 Minimal Estimator — 一个 fit + 一个 predict 就够了 (v0.1).

Run:    python s01_minimal_estimator/code.py
"""
# --- v0.1.1: training state + fit/predict lifecycle ---
from __future__ import annotations
import numpy as np

def _check_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got ndim={y.ndim}")
    return y

class MeanRegressor:
    """Predicts the training target mean for every sample.

    Demonstrates the minimal Estimator lifecycle: fit stores learned
    parameters (trailing-underscore attributes), predict reads them.
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = _check_1d(y)
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have same length")
        self.constant_ = y.mean()
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "constant_"):
            raise RuntimeError("This MeanRegressor is not fitted yet.  Call 'fit' first.")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")
        return np.full(X.shape[0], self.constant_)


class MostFrequentClassifier:
    """Predicts the most frequent training class for every sample."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = _check_1d(y)
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have same length")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.most_frequent_ = self.classes_[np.argmax(counts)]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "most_frequent_"):
            raise AttributeError("Not fitted yet. Call 'fit' first.")
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[0], self.most_frequent_)


def main() -> None:
    # synthetic: y ≈ 3.5 with noise
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 2))
    y = 3.5 + 0.2 * rng.standard_normal(20)

    reg = MeanRegressor()
    reg.fit(X, y)
    preds = reg.predict(X)
    score = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)

    print(f"Train mean: {reg.constant_:.2f}")
    print(f"Predictions: {preds[:3]}")
    print(f"R²: {score:.2f}")
    assert reg.predict(X).shape == (20,)
    assert np.allclose(reg.predict(X[:2]), reg.constant_)

    # classifier
    Xc = rng.standard_normal((20, 2))
    yc = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1] * 2)
    clf = MostFrequentClassifier().fit(Xc, yc)
    assert np.all(clf.predict(Xc) == clf.most_frequent_)
    print("Most-frequent class:", clf.most_frequent_)
    print("[PASS] s01 minimal estimator: fit→predict 闭环 + `_` 后缀训练状态正确")

if __name__ == "__main__":
    main()
