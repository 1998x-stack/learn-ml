import numpy as np
from src.mini_sklearn.model_selection import train_test_split


def test_train_test_split_shapes_and_content():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    # 返回顺序：所有训练数组在前，然后所有测试数组在后
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    assert X_tr.shape == (15, 2) and X_te.shape == (5, 2)
    assert y_tr.shape == (15,) and y_te.shape == (5,)
    # 训练/测试是原数据的一个划分：并集覆盖全部行（用 y 校验最简单）
    assert np.sort(np.concatenate([y_tr, y_te])).tolist() == np.sort(y).tolist()


def test_train_test_split_deterministic_across_calls():
    a = train_test_split(np.arange(100.0), np.arange(100), random_state=7)
    b = train_test_split(np.arange(100.0), np.arange(100), random_state=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_train_test_split_rejects_invalid_sizes():
    import pytest

    for size in (0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.25"):
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(np.arange(10), test_size=size, random_state=0)


def test_train_test_split_rejects_empty_partition():
    import pytest

    for size in (0.1, 0.9):
        with pytest.raises(ValueError, match="at least one training"):
            train_test_split(np.arange(1), test_size=size, random_state=0)


def test_train_test_split_keeps_both_partitions():
    train, test = train_test_split(np.arange(2), test_size=0.5, random_state=0)
    assert len(train) == len(test) == 1
    assert sorted(np.concatenate([train, test]).tolist()) == [0, 1]
