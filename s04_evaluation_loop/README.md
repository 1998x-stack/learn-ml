# s04：Evaluation Loop——训练、预测、评估形成闭环

[返回根目录](../README.md)

> 本章口号：**没有独立测试集，就没有可信的模型效果。**
>
> 对应 sklearn 模块: `sklearn.model_selection`、`sklearn.metrics`、`sklearn.datasets`

## 1. 上一章留下的问题

前三章（s01–s03）解决了"模型怎么学、怎么校验输入、怎么统一协议"。但我们仍然回答不了最核心的问题——**模型到底好不好？**

```python
model.fit(X_train, y_train)
pred = model.predict(X_test)
# 拿训练集上的误差当真？—— 过拟合下数字好看，但不可信
print("MSE on training:", mean_squared_error(y_train, model.predict(X_train)))
```

把误差算在**训练过的那批数据**上，等于让考生做自己出题的试卷。真正的评估必须把一部分数据**留出来，训练时绝不让模型看见**——这就是独立的测试集。

## 2. 本章目标

读完本章，你应该能回答：

- 为什么必须有一个"训练时从未见过"的独立测试集？
- `train_test_split` 为什么用共享的索引置换，而不是分别随机切训练 / 测试？
- MSE / MAE / R² 各自刻画误差的哪个方面？R² 的分母为什么是"猜均值"？
- accuracy 什么时候是诚实指标，什么时候会骗人（类别不平衡）？
- `make_regression` / `make_classification` 如何零成本造出可复现的合成数据？

核心思想：**评估必须闭环**。`X, y → 切分 → 训练 → 预测 → 指标`，每一步都串起来，才得到一个可信的效果数字。

## 3. 数学直觉

评估的直觉是"防作弊"：测试集就是一场**闭卷考试**，训练集是开卷复习。

```text
训练集（一道题）：fit 阶段可见，模型背答案也随它去
测试集（另一道题）：fit 阶段从不相见，predict 阶段才揭卷
指标 = 试卷得分：    MSE / accuracy 等，在测试集上算出
```

`train_test_split` 的"共享置换"保证训练与测试是同一个数据集的**互斥划分**：每一行要么进训练、要么进测试，绝不重复。否则若两边都有第 k 行，评估就是"开卷考试"。

## 4. 数学定义

本章新增四组构件：划分、回归指标、分类指标、合成数据。

```text
train_test_split(X, y, test_size=0.25, random_state=None) -> (X_tr, y_tr, X_te, y_te)
    共享索引置换切分；所有训练数组在前，测试数组在后

MSE  = mean((y - y_pred)^2)                    # 均方误差
MAE  = mean(|y - y_pred|)                      # 平均绝对误差
R²   = 1 - sum((y-y_pred)^2) / sum((y-mean(y))^2)   # 决定的系数
acc  = mean(y == y_pred)                       # 分类准确率
```

R² 的分母 `sum((y-mean)^2)` 是"拿均值瞎猜"的误差，分子是模型的误差——所以 R² 衡量模型**比均值猜测好多少**：1 完美、0 和均值持平、负值比均值还差。

## 5. 最小实现

先看最朴素的"共享置换切分"——一次 `permutation` 同时打动两条路径：

```python
def _split(X, y, test_size=0.25, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    perm = rng.permutation(n)               # 一个共同的随机排列
    n_test = int(np.ceil(n * test_size))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
```

评估就是三个加在一起：

```python
X, y = make_regression(random_state=42)          # 造数据
X_tr, y_tr, X_te, y_te = _split_test(X, y)       # 切数据（独立测试集）
pred = MeanRegressor().fit(X_tr, y_tr).predict(X_te)  # 训练 + 预测
score = _r2(y_te, pred)                          # 在测试集上算分
```

## 6. NumPy 向量化实现

所有指标都是一条向量化表达式，零 for 循环：

```python
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))
acc = np.mean(y == y_pred)

def r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot
```

合成数据同样向量化。回归是线性生成：`y = X @ coef + noise * randn`。分类把每个类别做成一个高斯团——给每个类别一个随机中心，样本沿该中心加噪：

```python
X = rng.standard_normal((n_samples, n_features))
centers = rng.normal(size=(n_classes, n_features))
classes = rng.integers(0, n_classes, size=n_samples)
X = X + centers[classes]   # 每个类别的样本围绕自己的中心聚集
```

## 7. scikit-learn 风格封装

框架轨道把四组能力放进 `src/mini_sklearn/` 的三个子包：

```python
from src.mini_sklearn.datasets import make_regression, make_classification
from src.mini_sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from src.mini_sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
pred = MeanRegressor().fit(X_tr, y_tr).predict(X_te)
print(r2_score(y_te, pred))
```

教学轨道的 `train_test_split` / `mean_squared_error` 内联在 `code.py` 里自包含，行为与框架轨道一致，仅省略了多数组验长度、`check_random_state` 归一化等工程细节。

## 8. 相对上一章的变化

上一章（s03）建立了"入口守卫"——坏数据在 `fit` / `predict` 前就被拦截。

本章补上"结果评估"：

- s01–s03：只会 `fit` / `predict`，从不问"效果数字是多少"——模型好坏只能靠肉眼。
- s04：新增 `train_test_split` 造出可信的测试集，新增回归 / 分类指标量化误差，新增 `make_*` 让数据可复现。

`confusion_matrix`（分类）是第一个"结构型"输出：不只一个标量，而是 `(k, k)` 的计数矩阵，第 `i` 行第 `j` 列 = 真实类 `i` 被预测成 `j` 的次数。`labels` 参数用 `np.unique(concat(y_true, y_pred, labels))` 把**没出现的类别也补成零行**，让矩阵形状稳定。

## 9. 运行实验

```bash
# 教学轨道：自包含可运行（回归 + 分类两个评估闭环）
python s04_evaluation_loop/code.py

# 框架轨道：pytest 验证划分、指标、数据生成
python -m pytest tests/model_selection/ tests/numerical/ -q
```

教学轨道示例输出（种子 `np.random.default_rng(42)`）：

```text
=== 回归 ===
MSE: 4.7946
MAE: 1.7575
R² : -0.0453
=== 分类 ===
accuracy: 0.3750
[PASS] s04: 独立测试集 + 评估闭环 —— 没有独立测试集就没有可信效果
```

测试预期输出：

```text
14 passed in 0.12s
```

## 10. 与 sklearn 对比

框架轨道是 `sklearn.model_selection` / `sklearn.metrics` / `sklearn.datasets` 三个模块的缩小版：

```python
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.datasets import make_regression, make_classification
# 本章
from src.mini_sklearn.model_selection import train_test_split
from src.mini_sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from src.mini_sklearn.datasets import make_regression, make_classification
```

关键差异（我们刻意简化）：

- sklearn 的 `train_test_split` 还有 `stratify`（分层抽样）、`shuffle`、`train_size` 等工程选项；本章只保留 `test_size` + `random_state`。
- sklearn 的 `make_classification` 有 `weights`、`n_informative`、`n_redundant`、`class_sep` 等一整套生成控制；本章简化成"高斯团 + 均匀类别"。
- sklearn 的 `confusion_matrix` 还用 `labels` 顺序精确对齐类别索引，并有 `sample_weight`；本章用 `np.unique` 合并顺序。

## 11. 失败案例和边界条件

- **把指标算在训练集上**：训练集上 MSE 几乎总小于测试集（过拟合击穿）；significance 在独立测试集上才算数。
- **测量集未独立**：若 `X_te` 与 `X_tr` 有重叠行，评估就"开卷"了；`train_test_split` 的共享置换保证互斥。
- **R² 的零方差分母**：当 `y` 恒定（`ss_tot == 0`）时除法无定义；框架轨道用 `ss_res == 0` 返回 1.0，否则返回 `NaN`。
- **accuracy 的失衡陷阱**：类别不平衡下，多数类正确率会虚高；`confusion_matrix` 能看清每一格。

## 12. 教学实现与生产实现的差距

- 教学的指标只算 `float` 标量；sklearn 的 `mean_squared_error` 支持 `multioutput`、`squared`（RMSE）等，并统一走 `_check_reg_targets` 校验。

- 教学的 `make_regression` / `make_classification` 是裸的向量化生成；sklearn 的实现包含庞大的、"结构化"噪声注入、特征相关控制与 `n_redundant` 的工程细节。

- 教学的 `train_test_split` 只处理数组；sklearn 还支持 `stratify`（分层抽样）、`shuffle`、`return_indices` 等完整生态。

这些仍是"教学优先"的取舍——评估的语义（切分、MSE、R²、accuracy）正确是核心，工程选项留给正式 sklearn。

## 13. 练习题

1. 把 `train_test_split` 的切分改成"分别随机采样两组"（不共享置换），然后断言两组可能出现重叠——观察它对"可信评估"的破坏。
2. 用训练集 `X_tr,y_tr` 算 `model.predict(X_tr)` 的 MSE，再算测试集上的 MSE，对比两者谁更小，思考为何。（过拟合）
3. 把 `r2_score` 的 `ss_tot` 换成一个很小的常数（比如 `1e-12`），观察对同一个回归输出 R² 会变为巨大的负值——体会分母"总方差"的意义。
4. 构造一个 `y` 全是同一类的数据喂给 `accuracy_score` 一个多数类分类器，对照 `confusion_matrix`，解释为什么 acc 可能虚高。

## 14. 下一章

现在我们能：造数据、切测试集、训练、评估（MSE/MAE/R²/accuracy）——**最小监督学习闭环（v0.1）至此形成**。

下一章进入 v0.2，从 `s05` 开始引入**预处理与线性学习**：标准化（StandardScaler）与线性模型（线性回归 / 梯度下降），学"可用任何特征组合推出可解释系数"的那套参数估计。
