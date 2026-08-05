# s06：Linear Regression——从均值预测到线性关系

[返回根目录](../README.md)

> 本章口号：**数据不再只决定一个常数，而是决定一组参数。**
>
> 对应 sklearn 模块: `sklearn.linear_model`

## 1. 上一章留下的问题

s01–s05 我们学会了规整数据，但所有"预测"本质上都返回**一个常数**：分类器给一个类别、Transformer 改一个状态、s04 的评估把分数的期望压成一个数字。

真实问题不是这样。身高和体重的关系、房价和面积的关系——它们是一条**线**，需要学**一组参数**（斜率 + 截距），而不是一个常数。s06 引入第一个真正的学习器：**线性回归**。它也是 s04 评估闭环里那个"训练"步骤第一次真正地去"学习"。

## 2. 本章目标

读完本章，你应该能回答：

- 线性回归"学了什么"？为什么 `coef_`、`intercept_` 末尾带下划线而 `fit_intercept` 不带？
- 闭式解的两种写法（正规方程 vs 伪逆 `np.linalg.pinv`）差别在哪？rank-deficient 时谁还能解？
- `fit_intercept=True` 时怎么把截距"吸收继承"进增广权重向量？
- `Ridge` 的 `alpha*I` 项解决了什么数值问题？

核心思想：**从"预测一个常数"升级为"预测一组参数"。**

## 3. 数学直觉

线性回归的直觉是：目标 `y` 是特征 `X` 的一个**线性组合**。

```text
没有学习器时   y_pred = mean(y)            # 一个常数，对所有人一样
有了线性回归   y_pred = X @ coef_ + intercept_  # 一组参数，随特征而变
```

`fit` 就是在解方程 `X w = y`：找一组权重 `w`，让 `X @ w` 尽可能接近 `y`。最小二乘损失 `||X w - y||^2` 的最优解就是闭式解——一步算完，不用梯度下降（梯度下降留给后续章节）。

## 4. 数学定义

```text
线性模型         y_pred = X @ coef_ + intercept_          # coef_ 形状 (n_features,)
无截距时         y_pred = X @ coef_                       # intercept_ = 0

最小二乘目标     min_w ||X w - y||^2

正规方程         w = (X^T X)^{-1} X^T y                   # 需要 X^T X 可逆
伪逆             w = pinv(X) @ y                          # 任何 X 都能解

Ridge            w = (X^T X + alpha I)^{-1} X^T y         # alpha>0 惩罚系数的模
```

`fit_intercept=True` 时，把 `X` 前面拼一列 1（`_append_ones`），解增广系统得到 `w_aug = [intercept, coef...]`，再拆回 `intercept_` 与 `coef_`。

## 5. 最小实现

闭式解最小实现极其短。单特征时可写成手算闭式解：

```python
slope     = sum((x- x_bar)*(y - y_bar)) / sum((x - x_bar)**2)
intercept = y_bar - slope * x_bar
```

多特征（含截距）用一枚伪逆就够：

```python
def fit(self, X, y):
    Xa = np.hstack([np.ones(len(X)), X]) if self.fit_intercept else X
    w = np.linalg.pinv(Xa) @ y
    self.intercept_ = w[0] if self.fit_intercept else 0.0
    self.coef_ = w[1:] if self.fit_intercept else w
    return self
```

伪逆 `pinv` 是"最广谱"的闭式解：矩阵奇异、列线性相关、样本少于特征，它都给你一个最小范数最小二乘解。

## 6. NumPy 向量化实现

整章零 for 循环：

```python
# 增广：把截距吸收成"1 那一列的系数"
Xa = np.hstack([np.ones((n, 1)), X])

# 伪逆最小二乘（等价于 pinv(Xa) @ y）
w = np.linalg.pinv(Xa) @ y

# 正规方程（X^T X 可逆时等价）
w = np.linalg.solve(Xa.T @ Xa, Xa.T @ y)

# 预测
y_pred = X @ coef_ + intercept_
```

教学轨道特意用了一列与自身重复的 `Xdup`（`X^T X` 奇异）来演示：正规方程在 `np.linalg.solve` 会爆，而 `pinv` 照样给出合理系数——这就是 sklearn 内部选 `pinv/lstsq` 的原因。

## 7. scikit-learn 风格封装

框架轨道放进 `src/mini_sklearn/linear_model/`，与 s04 指标体系同构：

```python
from src.mini_sklearn.linear_model import LinearRegression, Ridge

lr = LinearRegression(fit_intercept=True).fit(X_train, y_train)
print(lr.coef_, lr.intercept_)        # 学习到的参数，带 _ 结尾
y_pred = lr.predict(X_test)

rg = Ridge(alpha=1.0).fit(X_train, y_train)
```

遵循统一协议：`fit` 返回 `self`、`coef_`/`intercept_`/`n_features_in_` 带 `_` 结尾、`get_params` 只返回超参数（`fit_intercept` / `alpha`）、未 `fit` 就 `predict` 报 `ValueError`。`_base.py` 里的 `_check_input`、`_add_ones` 被 `LinearRegression` 与 `Ridge` 共享。

## 8. 相对上一章的变化

上一章（s05）Transformer 学习"状态"但**不预测目标**。本章跨过这一步：

- s01–s05：模型给出常量输出（类别 / 状态 / 均值）。
- s06：`fit(X, y)` 真正解线性方程组，产出 `coef_` 与 `intercept_`，`predict` 吃特征出连续目标。
- 学习结果从"s向量/标量状态"变为"一组可解释系数"。
- 新增回归闭环：`切分 → fit(线性) → predict → R²`。

一个常见的误判是"线性回归就去调 `LinearRegression`"。本章正在拆的正是 `fit` 内部到底解了什么方程。

## 9. 运行实验

```bash
# 教学轨道：自包含可运行（手算闭式 + pinv 多特征 + 奇异列）
python s06_linear_regression/code.py

# 框架轨道：pytest 验证 numerical + api
python -m pytest tests/numerical/test_linear_regression.py tests/api/test_api_linear.py -q
```

教学轨道示例输出（种子 `np.random.default_rng(42)`）：

```text
=== 1) 单特征手算闭式解（先看简单的）===
手算 slope=1.000000, intercept=1.000000  (期望 1.0 / 1.0)
pinv coef_=[1.], intercept_=1.000000
R^2 = 1.0000

=== 2) 多特征 + 噪声：伪逆（pinv）===
coef_  [ 1.991 -0.998  0.504]   (真值 [ 2.  -1.   0.5])
intercept_ 2.984   (真值 3.0)
R^2 = 0.9981

[PASS] s06: 线性回归闭式解——手算 slope、pinv 多特征、奇异列落于 rank-deficient 仍可解
```

测试预期输出：

```text
16 passed in 0.11s
```

## 10. 与 sklearn 对比

框架轨道是 `sklearn.linear_model` 的缩小版：

```python
# sklearn
from sklearn.linear_model import LinearRegression, Ridge
# 本章
from src.mini_sklearn.linear_model import LinearRegression, Ridge
```

关键差异（我们刻意简化）：

- sklearn 的 `LinearRegression` 会跑到 `lstsq` / SVD 并支持稀疏；本章只做稠密 `pinv`。
- sklearn 的 `Ridge` 默认 `solver="auto"`，且把截距分离（对中心化数据解，截距不再正则化）；本章对增广矩阵整块解，截距也被 Regular（教学上更直接）。
- sklearn 支持 `n_features_in_` 之外的 `feature_names_in_` 与 `copy_X`；本章只留超参数 `fit_intercept`/`alpha`。

## 11. 失败案例和边界条件

- **未 `fit` 就 `predict`**：`coef_`/`intercept_` 不存在，应抛 `ValueError`（复用 `check_is_fitted`）；API 测试 `test_predict_before_fit_raises_value_error` 锁住。
- **正规方程在奇异矩阵崩**：`X^T X` 不可逆（重复列 / 特征数≥样本数）→ `np.linalg.solve` 报错。解：用 `pinv`（最小范数最小二乘），教学 code.py 用两列相同的 `X` 证明。
- **`set_params` 未知参数**：基类 `BaseEstimator.set_params` 只有 `hasattr` 成立才接收，否则抛 `ValueError`（API 测试锁定 `Invalid parameter`）。
- **特征数不匹配**：`predict(X)` 列数 ≠ `fit` 的 `n_features_in_`，应报明确错误；numerical 测试 `test_predict_rejects_wrong_feature_count`。
- **`alpha` 过小 + 强共线性**：`Ridge(alpha≈0)` 退化成普通最小二乘，数值不稳定；测试用 `alpha=1.0` 验证共线目录可解。

## 12. 教学实现与生产实现的差距

- 教学 `LinearRegression` 始终 `pinv`；sklearn 按 recond 切换 `lstsq`/SVD 且支持稀疏、多输出 `y`。
- 教学 `Ridge` 用 `np.linalg.solve` 直接解正规方程；sklearn 对共线与数值不稳定用 Cholesky/SVD/esolver 并默认切中心化。
- 教学只处理稠密 `float` 二维 `X` 与一维 `y`；sklearn 有 `multi_output`、`feature_names`、`sample_weight` 等。

这些仍是"教学优先"取舍——闭式解语义正确是核心，工程强度和算法鲁棒交给正式 sklearn。

## 13. 练习题

1. 在 `LinearRegression.fit` 前直接调用 `predict`，断言抛 `ValueError` 并检查错误信息含 "fitted"。
2. 构造两列完全相同（`np.column_stack([x, x])`）的 `X`，对比 `np.linalg.solve(X^T X, ...)` 与 `np.linalg.pinv(X) @ y`——前者为何失败？
3. 给 `Ridge` 设 `alpha=0.0` 与 `alpha=1e6`，观察 `coef_` 的模如何随惩罚变化；解释"大惩罚 → 小系数"。
4. 用 `np.random.default_rng` 生成三特征数据集，把真值改成 `[1, 2, 3]`，验证 `coef_` 收敛到真值附近，`R²` 接近 1。

## 14. 下一章

现在我们能：把连续目标拟合为一组线性参数，闭式求解并正确报告 `coef_`、`intercept_`、`R²`，且用 `pinv` 应对 rank-deficient。

下一章进入 `s07`：当 `X^T X` 不可逆或特征爆炸到 memory，闭式解力不从心——引入**梯度下降**，用迭代优化逼近同一组参数，打开高维与非线性学习的大门。
