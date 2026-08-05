import numpy as np
import pytest

from src.mini_sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def test_mean_squared_error_formula():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 3.0])
    # 手算： ((0)^2 + (-2)^2 + 0^2) / 3 = 4/3
    expected = (0.0 ** 2 + (-2.0) ** 2 + 0.0 ** 2) / 3
    assert mean_squared_error(y_true, y_pred) == pytest.approx(expected)


def test_mean_absolute_error_formula():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 3.0])
    # 手算：(|0| + |-2| + |0|) / 3 = 2/3
    expected = (0.0 + 2.0 + 0.0) / 3
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(expected)


def test_r2_score_perfect_prediction_is_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert r2_score(y_true, y_pred) == pytest.approx(1.0)


def test_r2_score_formula():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5])
    # 手工：ss_res = sum((y_true-y_pred)^2), ss_tot = sum((y_true-mean)^2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    expected = 1.0 - ss_res / ss_tot
    assert r2_score(y_true, y_pred) == pytest.approx(expected)


def test_r2_score_guards_zero_variance():
    # 目标恒定时 ss_tot == 0，r2_score 必须返回定义值而非除零
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    score = r2_score(y_true, y_pred)
    assert np.isfinite(score)


def test_accuracy_score_formula():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    # 匹配 4/5 = 0.8
    assert accuracy_score(y_true, y_pred) == pytest.approx(4 / 5)


def test_confusion_matrix_shape_and_content():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 1, 2])
    cm = confusion_matrix(y_true, y_pred, labels=np.array([0, 1, 2]))
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
    # 对角线为正确分类数
    expected = np.array(
        [
            [1, 0, 1],  # true=0: pred 0,2 -> (0,0)=1, (0,2)=1
            [0, 2, 0],  # true=1: pred 1,1 -> (1,1)=2
            [0, 1, 1],  # true=2: pred 1,2 -> (2,1)=1, (2,2)=1
        ]
    )
    assert np.array_equal(cm, expected)


def test_confusion_matrix_handles_missing_class():
    # test 数据没出现某个类别时，labels 仍保证该类别有对应行/列（全 0）
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=np.array([0, 1, 2]))
    assert cm.shape == (3, 3)
    assert np.array_equal(cm[2, :], np.zeros(3))


def test_confusion_matrix_derives_labels_from_data_by_default():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 2, 1])  # 2 出现在 y_pred 而未出现在 y_true
    cm = confusion_matrix(y_true, y_pred)
    # 类序按 np.unique(concat([y_true, y_pred]))：{0,1,2} -> (3, 3)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
    # 2 仅作为预测类别出现 -> 行全 0、列非零
    assert np.array_equal(cm[2, :], np.zeros(3))
    assert cm[:, 2].sum() == 1
