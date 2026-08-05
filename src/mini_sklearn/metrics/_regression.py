"""Regression metrics (s04 version)."""
from __future__ import annotations

import numpy as np

from ..utils.validation import check_consistent_length


def _check_y_true_pred(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    check_consistent_length(y_true, y_pred)
    return y_true, y_pred


def mean_squared_error(y_true, y_pred) -> float:
    """Mean squared error: sum((y_true - y_pred)^2) / n."""
    y_true, y_pred = _check_y_true_pred(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error: mean(|y_true - y_pred|)."""
    y_true, y_pred = _check_y_true_pred(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination: 1 - ss_res / ss_tot, guarded for zero variance."""
    y_true, y_pred = _check_y_true_pred(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        # 目标恒定（零方差）：完美预测（ss_res==0）定义为 1.0，否则该指标失去意义
        return float(1.0) if ss_res == 0.0 else float("nan")
    return float(1.0 - ss_res / ss_tot)
