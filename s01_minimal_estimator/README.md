# s01：Minimal Estimator——一个 `fit` 加一个 `predict` 就够了

[返回根目录](../README.md)

> 本章口号：一个 `fit` 加一个 `predict`，就是最小可运行模型。
>
> 对应 sklearn 模块: `sklearn.dummy`, `sklearn.base`

## 1. 问题

上一章的引子（这里是从「无」开始的）：模型「训练」到底保存了什么？

当我们调用 `LinearRegression().fit(X, y)` 之后，那个模型对象里究竟装进了什么东西？
为什么训练之后就能对没见过的新样本作出预测？训练和预测之间，靠什么衔接？

本系列的第一个问题，就是这个最小问题：**一次 `fit` 到底往模型里存了什么**。

## 2. 本章目标

读完本章，你应该能回答：

- 模型训练到底保存了什么？
- 超参数和训练参数有什么区别？
- 为什么训练属性使用 `_` 结尾？
- 为什么 `fit()` 返回 `self`？
- 为什么 s01 不去做「真正的」预测，而是先教会「基线模型」？

我们采用 sklearn 的 Estimator 生命周期约定——`fit(X, y) → predict(X)`——用两个最平凡的模型（回归器与分类器）把「训练状态」这个概念立起来。

## 3. 数学直觉

一个最「偷懒」的回归器：**永远预测训练集目标值的均值**。

直觉上它没有任何预测能力，因为所有样本的答案都一样。但它的价值不在正确率，而在于它里面有「一个数」——这个数是从训练数据里学出来的。

```text
训练 = 把数据压缩成参数
预测 = 用参数对所有样本回答
```

## 4. 数学定义

**MeanRegressor（均值回归器）**

用训练目标均值 `c` 作为对所有样本的常值预测。

```text
c = mean(y_train)
ŷ_i = c,  对于所有样本 i
```

**MostFrequentClassifier（众数分类器）**

用训练集中出现次数最多的类别 `m` 作为对所有样本的常值预测。

```text
c_m = argmax_k count(y_train == k)
y_hat_i = c_m,  对于所有样本 i
```

## 5. 最小实现

`fit` 只做一件事：把训练结果写进 `_` 结尾的属性；`predict` 只做一件事：读取这些属性并返回答。

```python
class MeanRegressor:
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
```

要点：

- **训练产物是参数 `_`-后缀**：`constant_`、`most_frequent_`、`n_features_in_`。
- **`fit` 返回 `self`**：方便链式调用 `MostFrequentClassifier().fit(X, y)`。
- **`predict` 前检查 `hasattr`**：未训练就预测会显式报错，而不是返回垃圾。

## 6. NumPy 向量化实现

与其写成逐样本 for 循环，不如让 NumPy 直接「广播」整个值：

```python
return np.full(X.shape[0], self.constant_)
```

`np.full` 一次性生成长度等于样本数的常值数组，`np.asarray(X, dtype=float)` 保证输入先被规整为浮点矩阵。分类器用 `np.unique(y, return_counts=True)` + `np.argmax(counts)` 一次性统计并取出众数——这些都是「向量化」而非「打开循环」的写法，也是 `fit` 的代价。

## 7. scikit-learn 风格封装

我们把两个模型放进同一个「生命周期协议」里：`fit(X, y) → self` 与 `predict(X) → array`。

```python
reg = MeanRegressor().fit(X, y)      # fit 返回 self，可链式调用
preds = reg.predict(X)               # 读取 fitted 属性
score = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)  # R²
```

`n_features_in_` 与 sklearn 一致：记录训练时的特征数，预测时校验输入维度，防止「训练时 2 列、预测时 3 列」的静默错误。

## 8. 相对上一章的变化

上一章（s00）没有任何代码，只搭建全景与最小内核的心智地图。本章是**第一个真正可运行的模型**：

- 引入 `fit`/`predict` 协议，运行 `python s01_minimal_estimator/code.py`。
- 引入训练状态：`fit` 写入 `_` 后缀属性，`predict` 读取。
- 引入本地校验助手 `_check_1d`（真正的共享 `check_X_y`/`check_array`/`check_is_fitted` 会在 s03 数据契约一章登场，这里先自包含实现）

## 9. 运行实验

```bash
python s01_minimal_estimator/code.py
```

种子 `np.random.default_rng(42)` 保证结果确定。示例输出：

```text
Train mean: 3.52
Predictions: [3.52386528 3.52386528 3.52386528]
R²: 0.00
Most-frequent class: 1
[PASS] s01 minimal estimator: fit→predict 闭环 + `_` 后缀训练状态正确
```

## 10. 与 sklearn 对比

调用方式与 sklearn 的 `DummyRegressor`/`DummyClassifier` 完全对齐：

```python
# sklearn
from sklearn.dummy import DummyRegressor, DummyClassifier
model = DummyRegressor(strategy="mean").fit(X, y)   # 同样预测均值
# 本章
model = MeanRegressor().fit(X, y)
```

两者都遵循相同的 Estimator 协议（`fit→self`、`_` 后缀 learned 属性、`n_features_in_`）。sklearn 的 `check_is_fitted` 靠该属性是否带 `_` 判断是否已训练——本章用 `hasattr(self, "constant_")` 手工表达了同一思想，正是 `sklearn.base` 行为的缩小版。

## 11. 失败案例和边界条件

- **未训练就预测**：`predict` 前检查 `hasattr`，未 `fit` 直接报错——回归器若 `constant_` 不存在抛 `RuntimeError`，分类器若 `most_frequent_` 不存在抛 `AttributeError`。
- **`y` 不是一维**：`_check_1d` 对 `ndim != 1` 抛 `ValueError`。
- **`X` 与 `y` 长度不一致**：`len(y) != X.shape[0]` 抛 `ValueError("X and y must have same length")`。
- **预测阶段特征数量变化**：`predict` 校验 `X.shape[1] != n_features_in_` 时抛 `ValueError`。

这些「显式报错」正是生产级 `check_array`/`check_X_y`/`check_is_fitted` 的雏形——先把错误暴露在训练/预测时，而不是让错误悄悄蔓延。

## 12. 教学实现与生产实现的差距

本章的实现是为了**讲清训练状态**，与 `sklearn.base` 的正式实现仍有不小差距：

- 数据校验被硬编码在 `fit`/`predict` 内部（手写 `np.asarray` + `hasattr`），而生产实现把 `check_array`、`check_X_y`、`check_is_fitted` 抽成共享的 `utils.validation`，由每个模型统一调用。
- `fit` 前我们没有任何 `get_params` / `set_params` / `clone` 协议，模型无法被 GridSearch、Pipeline 等「框架层」统一管理（那是 s02 的机制）。
- 类型注解、对稀疏矩阵/DataFrame、特征名 `feature_names_in_`、只读 fitted 属性等工程细节都缺席。
- 本章用 `dtype=float` 强转输入做了「最简单的类型契约」，但真正的契约还包括数值检查、有限值检查与 shape 标准化。

这些都是「蓝图实现不是生产实现」的取舍——教学 `code.py` 只突出本章机制。

## 13. 练习题

1. 给 `MeanRegressor` 增加一个超参数 `offset`（预测值 = 均值 + offset），观察它属于 `_` 后缀还是不带 `_`（提示：它是「超参数」，应在 `__init__` 里设置、不带 `_`）。

2. 用 `DummyClassifier(strategy="most_frequent")` 复现 `MostFrequentClassifier`，对比两者行为差异。

3. 先不 `fit` 直接 `predict`，确认报错信息能解释「先 fit 再预测」。

1. 给 `MeanRegressor` 增加一个超参数 `offset`（预测值 = 均值 + offset），观察它属于 `_` 后缀还是不带 `_`（提示：它是「超参数」，在 `__init__` 里设置、不带 `_`）。

2. 用 `DummyClassifier(strategy="most_frequent")` 复现 `MostFrequentClassifier`，比较两者对你的类的行为差异。

3. 预测前先抛这个 `AttributeError`（先不 `fit` 直接 `predict`），确认报错信息能解释「先 fit 再预测」。

## 14. 下一章

现在模型能训练了，但每个模型都可能使用不同的接口。

怎样让所有模型保持一致？—— 下一章 `s02_estimator_protocol` 引入 `BaseEstimator`、`get_params` / `set_params`、`clone` 与 Mixins，把「统一接口」固化下来，让「新增模型，不改变调用方式」成为一个可复用的协议。
