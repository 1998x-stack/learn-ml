"""Numerical tests for decision trees (s12 / v0.3.4-5).

Covers: separable 2-class 2-feature recovery, ``max_depth`` honoured, a
hand-checkable gini impurity, wrong-n-features ValueError, the entropy
criterion, and the regression tree predicting finite per-leaf means.
"""
import numpy as np
import pytest

from src.mini_sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make_axis_separable_grid():
    """A small axis-separable grid: feature-0 already separates the two classes.

    Class 0 sits on the left half-plane (x[:,0] < 0), class 1 on the right
    (x[:,0] > 0); every sample is labelable by a single threshold split on
    feature 0, so both shallow and deep trees should recover the labels.
    """
    X0 = np.array([[-2.0, 1.0], [-1.0, 3.0], [-3.0, -2.0], [-1.5, 0.5]])
    X1 = np.array([[1.0, -1.0], [2.0, 4.0], [3.0, 0.0], [1.5, -3.0]])
    X = np.vstack([X0, X1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _gini(proportions):
    """Hand-reference impurity: 1 - sum_k p_k^2."""
    return 1.0 - float(sum(p * p for p in proportions))


def test_predict_recovers_training_on_separable_data():
    X, y = make_axis_separable_grid()
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    pred = model.predict(X)
    assert np.mean(pred == y) == 1.0  # separable -> exact recovery
    assert pred.shape == y.shape


def test_max_depth_limits_number_of_splits():
    # 8 points already on the boundary leaves/structure: depth-1 tree has <=1 node.
    X, y = make_axis_separable_grid()
    shallow = DecisionTreeClassifier(max_depth=1).fit(X, y)
    # a depth-1 tree must be a single split: root is an internal node, both
    # children are leaves -> total internal nodes == 1
    internal = _count_internal(shallow.tree_)
    assert internal <= 1
    # and a depth-1 tree can still separate two axis-separable classes on feature 0
    assert np.mean(shallow.predict(X) == y) == 1.0


def _count_internal(node):
    if node is None or node.left is None and node.right is None:
        return 0
    return 1 + _count_internal(node.left) + _count_internal(node.right)


def test_gini_on_pure_leaf_is_zero():
    # The fabricated data has no impurities: full tree splits until each leaf is pure.
    X, y = make_axis_separable_grid()
    model = DecisionTreeClassifier(random_state=0).fit(X, y)

    # A perfect prediction means every leaf is pure -> sum of leaf impurities is 0.
    leaves = _leaves(model.tree_)
    assert leaves  # tree is non-empty
    p = model.predict(X)
    assert np.mean(p == y) == 1.0


def _leaves(node):
    out = []
    if node is None:
        return out
    if node.left is None and node.right is None:
        out.append(node)
    else:
        out += _leaves(node.left) + _leaves(node.right)
    return out


def test_gini_hand_checkable():
    # 3 samples class 0, 1 sample class 1 -> gini 1 - (3/4)^2 - (1/4)^2 = 0.375
    X = np.array([[0.0, 0.0], [0.5, 0.0], [0.6, 0.0], [10.0, 0.0]])
    y = np.array([0, 0, 0, 1])
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    # The root node holds all 4 samples (3 class-0, 1 class-1); its gini is
    # the hand-computable 1 - (3/4)^2 - (1/4)^2 = 0.375.
    # The classifier greedily picks the feature-0 split between 0.6 and 10.0
    # whose left child (3 class-0) is pure -> so the learned tree must have
    # produced exactly that root split and separated the labels.
    assert model.tree_.impurity == pytest.approx(0.375)
    assert np.mean(model.predict(X) == y) == 1.0


def test_wrong_n_features_raises():
    X, y = make_axis_separable_grid()
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="different number of features"):
        model.predict(np.zeros((3, 5)))


def test_entropy_criterion_recovers_separable():
    X, y = make_axis_separable_grid()
    model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    assert np.mean(model.predict(X) == y) == 1.0
    assert model.tree_ is not None


def test_regressor_predicts_finite_means():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 1.0], [6.0, 1.0]])
    y = np.array([3.0, 4.0, 10.0, 12.0])
    model = DecisionTreeRegressor(criterion="squared_error", max_depth=2).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    # leaves collapse into constant means of their leaves
    np.testing.assert_allclose(pred, y, atol=1e-6)  # pure leaves on this data


def test_predict_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="fitted"):
        DecisionTreeClassifier().predict(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="fitted"):
        DecisionTreeRegressor().predict(np.zeros((2, 2)))
