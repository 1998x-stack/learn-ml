# s08：Logistic Regression——从得分到概率再到类别

[返回根目录](../README.md)

> 本章口号：**从得分到概率再到类别。**
>
> 对应 sklearn 模块: `sklearn.linear_model`

## 1. 上一章留下的问题

s06/s07 都在做**回归**：让模型预测一个连续数值（房价、温度）。但很多任务要的是**分类**——股票涨还是跌、邮件是垃圾还是正常、疾病有还是没有。你不能把线性回归的 `X @ coef + intercept_` 直接当分类用：它是个无上下的实数，而类别是个离散集合。

s08 的答案分三步：先把那个实数**得分**压成一个 `[0,1]` 区间里的**概率**（sigmoid），再把这概率交给一个带 `_` 的属性 `classes_`——**从得分到概率再到类别**。同样走梯度下降（沿用 s07 的框架），但目标不再是 MSE，而是**二元交叉熵**。

## 2. 本章目标

读完本章，你应该能回答：

- 为什么分类要用 sigmoid + 交叉熵，而不是把它当回归做 MSE 硬套？
- `decision_function`、`predict_proba`、`predict` 三者是什么关系？各自长什么形状？
- `sigmoid(z) = 1/(1+exp(-z))` 在 `z` 很大 / 很小时为什么会**数值溢出**，怎么修？
- 为什么交叉熵的梯度是 `(p - y)` 一条这么简单的式子？
- `{a,b}` 这类任意标签怎么统一进 0/1 的内部表示？（`classes_`）

核心思想：**模型输出概率，类别是概率的 `argmax`。**

## 数学直觉

逻辑回归把线性得分 `z = X @ coef + b` 塞进 sigmoid，把它"夹"进概率：

```text
p = sigmoid(z) = 1 / (1 + exp(-z))          # (0,1)，单调递增
z 越正 → p 越接近 1（更像 class_1）；z 越负 → p 越接近 0
```

类别则取概率较高的一侧：`[1-p, p]` 里 `argmax` 对应 `classes_[argmax]`。sigmoid 的判定面仍是 `z=0` 的平面，和线性回归的 `Xw+b` 同构——只是多了外层这道"压到概率"的变换。

## 数学定义

```text
得分      z_i = x_i @ coef + intercept_
概率      p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))
交叉熵    loss = -(1/n) Σ [ y_i log p_i + (1-y_i) log(1-p_i) ]
梯度      grad(w) = (1/n) Xaᵀ (p - y)          # 交叉熵对得分求导恰为 (p-y)
更新      w ← w - learning_rate * grad
信号      argmax([1-p, p]) -> classes_[argmax]
```

注意交叉熵对 `w` 的梯度里那个漂亮的 `(p - y)`：它就是残差（预测概率减真实标签）的平均再乘 `Xaᵀ`。换目标就换这一行——MSE 那套推导直接被抛到一边。

## 最小实现

一个稳定 sigmoid + 一条 loss + 一个更新循环：

```python
def sigmoid(z):
    pos = z >= 0
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))          # 大 |z| 时不溢出
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out

loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

for _ in range(max_iter):
    p = sigmoid(Xa @ w)
    grad = (1.0 / n) * (Xa.T @ (p - y))               # ⭐ 整个逻辑回归的梯度
    w = w - learning_rate * grad
```

## NumPy 向量化实现

整批 `prediction` 用一次 `@` 完成、`grad` 又是一次 `Xa.T @ (p - y)`，全程无 for（除了迭代循环）：

```python
intercept, coef = w[0], w[1:]

def decision_function(X): return X @ coef + intercept          # 得分，(n,)
def predict_proba(X):    p = sigmoid(decision_function(X))     # 概率，(n,2)
                         return np.column_stack([1 - p, p])
def predict(X):          return classes[np.argmax(predict_proba(X), axis=1)]
```

`predict_proba` 每一行恰好和为 1，`predict` 只是 `classes_[argmax]`——类别不是学出来的，是识别的概率里挑大的那个。

## scikit-learn 风格封装

框架轨道新增 `src/mini_sklearn/linear_model/_logistic.py`，提供 `LogisticRegression`，与 s06/s07 共享 `_check_input`/`_add_intercept` 协议：

```python
from src.mini_sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000, learning_rate=0.1, random_state=0).fit(X, y)
print(clf.coef_, clf.intercept_)      # 学习到的权重，带 _
print(clf.classes_)                   # 排序后的唯一标签
print(clf.decision_function(X))       # 原始得分，(n,)
print(clf.predict_proba(X))           # [P(class_0), P(class_1)]，(n,2)，行和=1
print(clf.predict(X))                 # classes[argmax]，(n,)
```

- `LogisticRegression(learning_rate=0.1, max_iter=1000, random_state=0, fit_intercept=True)`：交叉熵上的全批 GD，更新 `w -= lr*(1/n) Xaᵀ(p-y)`。
- `classes_ = np.unique(y)`：任意标签 `{a,b}` 也被规整，`predict` 回 label，`predict_proba` 每行表示 `[P(class_0), P(class_1)]`。
- `_sigmoid` 用 sign 分支避免 `exp` 上溢；loss 内对 `p` 做了 `np.clip` 以免 `log(-inf)` 打爆。
- `fit` 返回 `self`；`predict`/`predict_proba`/`decision_function` 都经 `check_is_fitted`，并呼应 sklearn 特征数校验。

## 相对上一章的变化

| | s06/s07（回归） | s08（二元分类） |
|---|-----|-----|
| 输出 | 连续实数 `X@coef_+intercept_` | 概率 `sigmoid(z)` ∈ (0,1) |
| Loss | `MSE` | 二元交叉熵 `CE` |
| 梯度 | `(2/n) Xaᵀ(Xa w - y)` | `(1/n) Xaᵀ (p - y)` |
| 预测 | `predict` 直接给数值 | `decision_function` → `predict_proba` → `predict` |
| 新增属性 | `coef_`, `intercept_` | 同前 + `classes_` |
| 模型类型 | `RegressorMixin` | `ClassifierMixin` |

s08 的准确率测试、概率行和与 `[0,1]`、以及 `classes_` 的排序标签都由 `tests/` 锁定。

## 运行实验

```bash
# 教学轨道：自包含可运行（sigmoid → 交叉熵 → 概率与类别）
python s08_logistic_regression/code.py

# 框架轨道：pytest 验证 numerical + api
python -m pytest tests/numerical/test_logistic.py tests/api/test_api_logistic.py -q
```

教学轨道示例输出（两块相距很远的 blob）：

```text
=== v0.2.6: sigmoid + 交叉熵 + GD，loss 应一路走低 ===
loss 0.6921 -> 0.0000 (2000 步下降)

=== v0.2.7: decision_function -> predict_proba -> predict ===
decision_function 形状 (120,)，得分区间 [-13.81, 13.57]
predict_proba 行和≈1: True，多数类概率中位数 1.000

[PASS] s08: sigmoid + 交叉熵，得分 → 概率（predict_proba）→ 类别（predict），训练准确率 1.000
```

强分离数据上 `predict_proba` 逼近 0/1，准确率逼近 1.0。

## 与 sklearn 对比

```python
# sklearn
from sklearn.linear_model import LogisticRegression
# 本章
from src.mini_sklearn.linear_model import LogisticRegression
```

- sklearn 的 `sklearn` `LogisticRegression` 支持 `penalty`、多类（softmax）、`class_weight`、`solver`、`C`（正则强度）等；本章只留纯交叉熵 + `learning_rate`/`max_iter` 的二元版本，多类留给 s09。
- sklearn 的 `predict_proba`/`decision_function`/`predict` 语义与本章一致（形状 `(n,2)`/`(n,)`/`(n,)`）；本章连 `classes_` 都开放，让读者看清标签编码。
- sklearn 用 `class_weight` 处理不平衡；本章在"失败案例"里把失衡的教训当边界讲，不承诺内置。
- 本章对概率只做 `np.clip` 安全层，不承诺 `log-loss` 的校准、`predict_proba` 的 calibration 等生产细节。

## 失败案例和边界条件

- **sigmoid 溢出**：当 `z` 很大时 `np.exp(-z)` 下溢成 0 → 求 `1/exp(-z)` 发散；`z` 很负时 `exp(-z)` 溢出成 `inf`。教学/框架都按 `sign(z)` 分支，把 `exp` 放在小的一侧。
- **`log(0)`**：当 `p` 被 sigmoid 压到 0 或 1 时 `log(p)` 得 `-inf`。loss 里先 `np.clip(p, 1e-15, 1-1e-15)`。
- **标签不是 0/1**：`classes_=np.unique(y)`，`y==classes[1]` 才拿来做 0/1 编码；`predict` 用 `classes_[argmax]` 映射回原始标签。
- **类别数 ≠ 2**：`_check_input` 后若 `classes_` 只有 1 个或超过 2 个，框架抛 `ValueError`——本章只做二元，多类留给 s09。
- **没 fit 就推断**：`predict`/`predict_proba`/`decision_function` 都触 `check_is_fitted` → `ValueError`；API 测试锁定。
- **特征数错**：`X` 列数与 `n_features_in_` 不符时抛明确 `ValueError`；numerical 测试锁定。

## 教学实现与生产实现的差距

- 教学只用**全批 GD** 且无正则；生产默认带 `C`（L2 正则的盛装版）、`solver` 切换、`max_iter` 自动学习率。
- 教学 `classes_` 只存两个标签；生产 `predict_proba` 多类为 `(n,k)`，用 `softmax` 而非 `sigmoid`。
- 教学 对概率只做数值夹取；生产还有 `class_weight`、`sample_weight`、`intercept_scaling` 与 poly 特征工程，来自 `sklearn.preprocessing`。
- 教学 `random_state` 只控制初始化/洗牌；生产用确定性 `RandomState` 并在 `GridSearch` 里复现。

## 练习题

1. 把 sigmoid 换成不稳定的 `1/(1+np.exp(-z))`，分别跑 `z=10` 与 `z=-10`，看哪边产生什么 `inf`/下标警告？为什么 `np.exp` 要分支？
2. 用 `learning_rate` 从 `0.5` 调到 `5.0`，`loss_history` 会怎样？在 `s08_logistic_regression/code.py` 里观测梯度是否震荡/发散。
3. 手动把所有 `y` 改成 0 全零（只有一类），`fit` 会报什么？确认 `classes_` 的" 二元"边界在哪处触发崩溃。
4. 改动 `classes_[argmax]` 方向——如果把输出换成 `classes_[argmin]` 会得到反的标签。试着解释为什么 `argmax` 是“最大概率 → 最可能是那一类”。
5. 比较 `cross_entropy` 与 `MSE` 在分类上：把第 160 行左右的 loss 换成 `mean((p-y)**2)`，看收敛后是否依然逼近 0/1？交叉熵在 `p-y` 大时给出更强梯度，是它选它的原因（`y log p` 的曲面更尖）。

## 下一章

s08 把线性模型从回归带到**二元**分类。s09 将把这条路径扩展到**多分类**（softmax / one-vs-rest），并把 `LogisticRegression` 与 `LinearRegression` 收进 `base.EstimatorImpl` 协议；再往后 s10 用真正的 `sklearn.linear_model` 做 parity 对比。