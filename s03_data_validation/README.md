# s03：Data Contract——先校验数据，再学习参数

[返回根目录](../README.md)

> 本章口号：**模型可以不同，输入契约必须相同。**
>
> 对应 sklearn 模块: `sklearn.utils.validation`、`sklearn.utils`

## 1. 上一章留下的问题

前两章（s01、s02）建立了 `fit(X, y) → self` 的最小闭环，并把超参数反射成统一协议 `get_params` / `set_params` / `clone`。但我们的 `fit` / `predict` 仍**不做任何输入校验**：

```python
model.fit([1, 2, 3], [1, 2])        # y 长度与 X 不一致 → 静默错位
model.predict([[1, np.nan]])       # 含 NaN → 底层 NumPy 模糊异常
model.predict(X_wrong_features)    # 特征数不对 → 直接算错
```

更糟的是，这些错误往往不是报在我们的代码里，而是以 `np.asarray` 或 `mean` 抛出的一堆与模型无关的晦涩异常，或干脆**静默返回垃圾**。调用者根本不知道自己是传错了形状、传了 NaN，还是忘了先 fit。

## 2. 本章目标

读完本章，你应该能回答：

- `check_array` 为什么要求 `X` 必须是二维、默认拒绝 NaN / Inf？
- `check_X_y` 如何同时校验 `X` 与 `y`，并保证两者样本数一致？
- `check_is_fitted` 为什么能拦住"没训练就 predict"？
- `check_consistent_length` 如何统一长度契约？
- `check_random_state` 为什么把随机过程收敛到统一的 `random_state` 约定？

核心思想：**数据是一份契约**。每个 Estimator 在 `fit` / `predict` 前都用同一套检查守卫自己的输入——模型可以不同，输入契约必须相同。

## 3. 数学直觉

"契约"不改变任何数值结果，但它改变**谁在什么时候发现错误**：

```text
无契约： 把 X 传错 → 深处 np.mean / np.dot 抛晦涩异常，甚至静默算出错误答案
有契约： 传入 check_X_y → 立刻抛出"是哪一维、什么性质"的明确 ValueError
```

契约把三个内隐藏的错误提前到入口点：

- 形状契约：`X` 必须 2D `(n_samples, n_features)`；训练与预测的特征数必须一致。
- 取值契约：`y` 可一维或二维；NaN / Inf 默认不允许。
- 状态契约：预测前必须已训练（`_`-后缀学习属性存在）；随机过程必须接受 `random_state`。

## 4. 数学定义

数据契约的五个字面量函数（与 sklearn 对齐）：

```text
check_array(X, *, dtype=float, ensure_2d=True, allow_nan=False) -> ndarray
    校验 X：二维、无 NaN/Inf，返回 float ndarray

check_X_y(X, y) -> (X, y)
    校验 X 合法，并保证 len(y) == X.shape[0]（样本数一致）

check_consistent_length(*arrays) -> True
    所有输入沿第一轴长度相同，否则 raise

check_is_fitted(estimator, attributes, msg=None) -> None
    断言 estimator 拥有全部 attributes（训练产物），否则 raise

check_random_state(seed) -> np.random.Generator
    None/int/Generator 统一转成 Generator；非法 seed raise
```

统一规则：

```text
X ：二维数组，形状 (n_samples, n_features)
y ：一维或二维目标
训练和预测的特征数量必须一致
NaN 和 Inf 默认不允许
随机过程必须支持 random_state
```

## 5. 最小实现

先看最朴素的二维校验——只检查 `ndim` 与 NaN/Inf：

```python
def _check_array(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got ndim={X.ndim}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or infinity; not allowed by default")
    return X
```

校验"长度一致"用一行集合去重判断：若所有数组第一轴长度都相同，长度集合只有一个元素。

```python
def _check_X_y(X, y):
    X = _check_array(X)
    y = np.asarray(y)
    if len(y) != X.shape[0]:
        raise ValueError(f"inconsistent lengths: X={X.shape[0]}, y={len(y)}")
    return X, y
```

## 6. NumPy 向量化实现

`check_array` 靠 `np.asarray` 统一类型，靠 `np.isnan(...).any()` / `np.isinf(...).any()` 一条向量化表达式扫完整张表，零 for 循环。

`check_consistent_length` 用集合推导对所有数组求长度，再判断是否只含一个长度：

```python
lengths = {len(np.asarray(a)) for a in arrays}
if len(lengths) > 1:
    raise ValueError(f"inconsistent lengths: {lengths}")
```

把校验接入 `MeanRegressor`，`fit` / `predict` 的第一句话就是守卫：

```python
class MeanRegressor:
    def fit(self, X, y):
        X, y = _check_X_y(X, y)          # 入口先校验
        self.constant_ = float(y.mean())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        _check_is_fitted(self, "constant_")  # 必须已训练
        X = _check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from fitted data")
        return np.full(X.shape[0], self.constant_)
```

## 7. scikit-learn 风格封装

框架轨道的正式实现放 `src/mini_sklearn/utils/validation.py`：

```python
from src.mini_sklearn.utils.validation import (
    check_array, check_X_y, check_is_fitted,
    check_consistent_length, check_random_state,
)

X = np.ones((10, 3)); y = np.ones(10)
X_c, y_c = check_X_y(X, y)      # 通过
check_is_fitted(model, ["constant_"])   # 未 fit 则 raise
rng = check_random_state(42)    # 统一随机过程
```

教学轨道的 `_check_array` / `_check_X_y` / `_check_is_fitted` 内联在 `code.py` 里自包含，行为与框架轨道一致，只是去掉了 `check_consistent_length` / `check_random_state` 的工程细节。

## 8. 相对上一章的变化

上一章（s01–s02）只有"协议"，没有"守卫"：模型会接受任何形状、任何取值。

本章把 `fit` / `predict` 变得**防御式**：

- s01/s02：`X = np.asarray(X, dtype=float)` 直接转，坏数据要么静默错位、要么抛晦涩异常。
- s03：任何状态都先走 `check_*`，非法输入抛出带中文标签的明确 `ValueError`。

框架轨道在 `src/mini_sklearn/` 下新增 `utils/` 子包，并新增 pytest 测试 `tests/api/test_validation.py` 对五个函数做机器验证——数据契约从此被测试守护，而不只靠读代码。

## 9. 运行实验

```bash
# 教学轨道：自包含可运行（含 4 个失败实验）
python s03_data_validation/code.py

# 框架轨道：pytest 验证五个校验函数
python -m pytest tests/api/test_validation.py -q
```

教学轨道示例输出（种子 `np.random.default_rng(42)`）：

```text
predict: [2.09445213 2.09445213 2.09445213 2.09445213]
rejected bad length: inconsistent lengths: X=2, y=3
rejected wrong n_features: X feature count differs from fitted data
rejected not-fitted: This MeanRegressor is not fitted yet.
rejected NaN: X contains NaN or infinity; not allowed by default
[PASS] s03: 数据契约统一 —— 非法输入产生明确错误，而非底层 NumPy 模糊异常
```

测试预期输出：

```text
7 passed in 0.11s
```

## 10. 与 sklearn 对比

框架轨道是 `sklearn.utils.validation` 的缩小版：

```python
# sklearn
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
# 本章
from src.mini_sklearn.utils.validation import check_array, check_X_y, check_is_fitted
```

关键差异（我们刻意简化）：

- sklearn 的 `check_array` 还处理 `accept_sparse`（稀疏矩阵）、`ensure_min_samples/features`、`copy` 拷贝、`dtype` 就近提升等一整套工程规则；本章只保留形状与取值范围。
- sklearn 的 `check_X_y` 有 `multi_output` / `y_numeric` / `ensure_all_finite`，支持同时校验多个 `y` 列；本章只做最基础的样本数一致。
- sklearn 的 `check_is_fitted` 会检查 `__sklearn_tags__` 与复数属性推断，并允许"没有训练但结构可预测"的例外；本章只做简单 `hasattr`。

## 11. 失败案例和边界条件

本章是"失败实验"章——故意喂错误数据，要求抛明确错误：

- **拟合时 `y` 长度不一致**：`MeanRegressor().fit([[1,2],[3,4]], [1,2,3])` → "inconsistent lengths: X=2, y=3"。
- **predict 特征数与训练不一致**：`model.predict(X[:, :2])` → "X feature count differs from fitted data"。
- **未 fit 就 predict**：`MeanRegressor().predict(X)` → "is not fitted yet"。
- **含 NaN 输入**：`model.fit(X_nan, y)` → "X contains NaN or infinity"。

边界说明：`np.isnan` 只对浮点有意义；若传入字符串/数值混排，`np.asarray(dtype=float)` 会先抛 `ValueError`，不走到 NaN 检查——这也是"契约前置"的表现。

## 12. 教学实现与生产实现的差距

- 教学的 `_check_array` / `_check_X_y` / `_check_is_fitted` 只有几十行；sklearn 的 `validation.py` 上千行，还承载 `check_random_state` 的整套生态。
- sklearn 校验支持稀疏矩阵、内存布局、输入 dtype 一致性、返回值是否复制，以及「校验后是否改变对象」等工程维度，教学轨道全部省略。
- `check_random_state` 生产上还处理旧式 `RandomState`，`np.random.seed` 全局种子时代被 `default_rng` 取代；教学只讲 `np.random.Generator` 一站式接口。
- 教学实现的错误信息为中文注释辅助的可读英文；生产实现面向库 API 的稳定英文信息。

这些都是「教学优先」的取舍——契约的正确性（形状、取值、训练态）是核心，工程健壮性留给正式 sklearn。

## 13. 练习题

1. 把 `MeanRegressor.fit` 里 `_check_X_y` 换成 s01 的裸 `np.asarray`，再跑 `code.py`，观察第 1、4 个失败实验是否还能抛"明确错误"。
2. 往 `_check_array` 传 1D 列表 `[1,2,3]`，确认 raise；再把 `ensure_2d=False` 接回框架 `check_array`，对比行为。
3. 把 `X_nan` 改成整数数组再 fit，观察 `np.isnan` on 整数 dtype 是否有警告/误判，思考为何教学默认 `float`。
4. `check_random_state` 传入字符串 `"abc"`，确认抛 ValueError；传入 `np.random`，验证返回的是可用的 `Generator`。

## 14. 下一章

现在每个 Estimator 在 `fit` / `predict` 前都会先经过 `check_*`，非法输入会在入口处得到明确错误。但训练完之后，**我们还没法评估模型好不好**——MSE / R² / accuracy 都没有，只能矛盾地"反正它预测了"。

下一章 `s04_evaluation_loop` 引入 `train_test_split` 与一组回归 / 分类指标，加上 `make_*` 合成数据，让「训练、预测、评估」真正闭环。
