[返回根目录](../README.md)

# s07：Optimization——迭代优化的三副面孔（GD / SGD / 学习率调度）


> 本章口号：**优化器负责寻找参数，模型负责定义目标。**
>
> 对应 sklearn 模块: `sklearn.linear_model`, `sklearn.utils.optimize`

## 1. 上一章留下的问题

s06 的 `LinearRegression` 用**闭式解**（`np.linalg.pinv`）一步算出 `coef_`/`intercept_`——漂亮，但苛刻：它要求我们能写出并求解 `X^T X` 的逆（或伪逆）。现实里这常常做不到：

- 特征成千上万，`X^T X` 太大、求逆慢到不可接受；
- 模型不是线性的（s10 之后），根本**没有闭式解**；
- 数据大到处在一个内存塞得下、一次梯度算不出来。

s07 抛出另一条路：**不做闭式解，用梯度迭代地"走下山"**。优化器（optimizer）只负责"怎么找参数"，与"要找哪组参数"（模型定义的目标 / loss）解耦。这正是本组口号要传达的分工。

## 2. 本章目标

读完本章，你应该能回答：

- 全批 `GD` 和随机 `SGD` 差在哪一步？为什么 `SGD` 逐 epoch 的 loss 会"抖动"而 GD 单调？
- 学习率（`learning_rate`）太大 / 太小各会怎样？为什么要做**调度** `lr / (1 + 0.01*iter)`？
- 梯度 `(2/n) X^T (X w - y)` 是怎么来的？（MSE 对 `w` 求导）
- 为什么 `SGD` 能扩展到闭式解和全量 GD 都扛不住的大数据？
- 解同一组参数，闭式 / GD / SGD 三个 `fit` 要怎么**收敛到同一份 `coef_`/`intercept_`**？

核心思想：**参数不是"算出来的"，而是被"找出来的"。**

## 数学直觉

记住一句话——梯度指向 loss 上升最快的方向，所以**走负梯度方向**会让 loss 变小：

```text
w  ←  w - learning_rate * grad        # 顺着"下坡"走一小步
grad = d(loss) / d(w)                  # loss 对 w 的偏导
```

`learning_rate`（步长）决定每次走多远。太小 → 磨蹭不动；太大 → 一步越过山谷，在两侧来回震荡甚至发散。

MSE loss 对增广权重 `w_aug = [intercept, coef...]` 的梯度非常好算：

```text
loss(w) = (1/n) ||Xa w - y||²
grad(w) = (2/n) Xaᵀ (Xa w - y)     # 全批：一次用全部样本
```

`SGD` 几乎一样，只是把每个样本的梯度单独更新一次：`grad_i = 2 * xa_i * (xa_i·w - y_i)`，再逐样本 `w -= lr*grad_i`。

## 数学定义

```text
全批 GD   grad = (2/n) Xaᵀ (Xa w - y)              w <- w - lr * grad
SGD       grad_i = 2 * xa_i * (xa_i@w - y_i)       每 epoch 洗牌，逐样本更新
调度       lr_eff = lr / (1 + 0.01 * iter)          大步到近旁，小步精修

闭式(对照) w = pinv(Xa) @ y                          一步到位，但要求可逆
```

`s_fit_intercept=True` 时，把 `X` 前面拼一列 1（复用 s06 的 `_add_intercept`），解出 `w_aug = [intercept, coef...]`，再拆回 `intercept_` 与 `coef_`——与 s06 完全同构。

## 最小实现

一支循环加一条更新就能描述整个 GD：

```python
Xa = np.hstack([np.ones((n, 1)), X]) if fit_intercept else X
w = np.zeros(Xa.shape[1])
for _ in range(max_iter):
    grad = (2.0 / n) * Xa.T @ (Xa @ w - y)      # 全批梯度
    w = w - learning_rate * grad
```

加一个 `rng.permutation` + 单样本更新，就变成 SGD：

```python
for i in rng.permutation(n):
    xi = Xa[i]
    w = w - lr * 2.0 * xi * (xi @ w - y[i])
```

## NumPy 向量化实现

整章只用一个 `@` 就能并行处理整批数据的梯度，无需 for 循环：

```python
# 更新方向 = Xaᵀ @ 残差（向量化），再按 n 平均
grad = (2.0 / n) * (Xa.T @ (Xa @ w - y))
w   -= learning_rate * grad
loss = float(np.mean((Xa @ w - y) ** 2))   # 记录当前 loss，画下降曲线
```

SGD 唯一的循环在"样本下标"（单样本步进必须逐个走），但每个样本的梯度仍是 `xi * (xi@w - y_i)` 的向量点积。教学 code.py 里我们刻意保存了 `loss_history_`，把三条曲线（GD / GD+decay / SGD）的下降打印出来对比。

## scikit-learn 风格封装

框架轨道新增 `src/mini_sklearn/linear_model/_stochastic_gradient.py`，提供 `GDRegressor` 与 `SGDRegressor`，与 s06 的 `LinearRegression`/`Ridge` 统一协议：

```python
from src.mini_sklearn.linear_model import GDRegressor, SGDRegressor

gd = GDRegressor(learning_rate=0.05, max_iter=2000).fit(X_train, y_train)
print(gd.coef_, gd.intercept_)                    # 学习到的参数，带 _
print(gd.loss_history_, gd.n_iter_, gd.converged_)

sgd = SGDRegressor(random_state=0, max_iter=2000).fit(X_train, y_train)
```

- `GDRegressor(learning_rate=0.01, max_iter=1000, tol=1e-4, fit_intercept=True)`：全批梯度，`grad = (2/n) Xaᵀ(Xa w - y)`，当 `‖grad‖ ≤ tol` 提前收敛并置 `converged_=True`。
- `SGDRegressor(learning_rate=0.01, max_iter=1000, random_state=0, fit_intercept=True)`：通过 `utils.validation.check_random_state` 生成可复现的洗牌 `rng`，每 epoch 单样本步进。
- 两者都输出 `coef_`、`intercept_`、`n_features_in_`、`loss_history_`、`n_iter_`；`fit` 返回 `self`；`predict` 复用 `check_is_fitted`/`check_array` 并校验特征数；`_base.py` 的 `_check_input`/`_add_intercept` 被 s06 与 s07 共享。

## 相对上一章的变化

| | s06 | s07 |
|---|-----|-----|
| 求解方式 | 闭式 `pinv(X) @ y` | 迭代 `w -= lr * grad` |
| `coef_`/`intercept_` | 一步解出 | 逐步逼近到同一组值 |
| 能应付特征爆炸/非线性 | 不能 | 能（只需 `grad` 可算） |
| 大数据 | 整体求解 | SGD 逐样本，扩展性好 |
| 产物 | `coef_`, `intercept_` | 同前 + `loss_history_`, `n_iter_`, `converged_` |

一个常见误判是"梯度下降就是要调 `SGDRegressor`"。本章拆的是**优化器与模型的分工**：同样的 MSE 目标，闭式、GD、SGD 三种"找参数"的办法收敛到同一份 `coef_`。

## 运行实验

```bash
# 教学轨道：自包含可运行（GD / 学习率调度 / 迷你 SGD）
python s07_optimization/code.py

# 框架轨道：pytest 验证 numerical + api
python -m pytest tests/numerical/test_sgd.py tests/api/test_api_sgd.py -q
```

教学轨道示例输出（种子 `np.random.default_rng(42)`）：

```text
=== 1) 全批梯度下降（无调度）===
[GD     ] coef_=[3.016] intercept_=2.008 (真值 3.0/2.0)  loss 16.803->0.0263 (190 步下降)  R²=0.998

=== 2) 学习率调度 lr/(1+0.01*iter)：大步到近旁，小步精修 ===
[GD+decay] coef_=[3.016] intercept_=2.008 (真值 3.0/2.0)  loss 16.803->0.0263 (200 步下降)  R²=0.998

=== 3) 迷你 SGD：单样本步进，噪声大但能处理超大语料 ===
[SGD    ] coef_=[2.984] intercept_=1.981 (真值 3.0/2.0)  loss 16.803->0.0284 (24 步下降)  R²=0.998

[PASS] s07: GD 全量步进、学习率调度、迷你 SGD——同一组参数三种找法
```

测试预期输出：`13 passed in ~1s`（加上 s06 的 58 项全绿为 `71 passed`）。

## 与 sklearn 对比

```python
# sklearn
from sklearn.linear_model import SGDRegressor
# 本章
from src.mini_sklearn.linear_model import SGDRegressor, GDRegressor
```

- sklearn 的 `SGDRegressor` 用 `X.T X` 相关的 `loss="squared_error"`，支持 `penalty`、`alpha`、`l1_ratio` 以及 `early_stopping`；本章只留纯 MSE + `learning_rate`/`max_iter`。
- sklearn 的 `learning_rate` 本身是字符串（`"constant"/"invscaling"/"adaptive"`）；本章直接给数值，把"调度"暴露成教学里手写的 `lr/(1+0.01*iter)`。
- sklearn 没有单独的 `GDRegressor`（全批 GD 由 `SGDRegressor(tol=...)` 化工）；本章拆出来讲清"全批 vs 随机"两种更新。
- 框架仍未实现动量、`n_iter_no_change` 的真正学习、稀疏矩阵等；留给更高版本。

## 失败案例和边界条件

- **`learning_rate` 太大**：loss 在起点震荡甚至发散，`loss_history_` 不降反升。教学用调度证明：大步伐开局、`lr/(1+0.01*iter)` 把步伐逐步收小，稳到收敛。
- **`learning_rate` 太小**：`converged_` 迟迟不触发，`max_iter` 跑完还在缓慢下降。lesson：学习率是"步伐"，不是"方向"。
- **`SGD` 的 loss 抖动**：逐样本更新带随机噪声，`loss_history_` 不是单调下降（教学 code.py 只断言"最终好于起点"），这正常——方差随着样本量的增多而退去。
- **未 `fit` 就 `predict`**：无 `coef_`/`intercept_`，`check_is_fitted` 抛 `ValueError`；API 测试锁定。
- **特征数不匹配**：`predict(X)` 列数 ≠ `n_features_in_`，应抛明确错误。
- 洗牌随机性依赖 `random_state`：不固定它就重现不了实验——这正是 `SGDRegressor(random_state=0)` 存在的理由。

## 教学实现与生产实现的差距

- 教学 只对线性回归做 MSE 的 GD/SGD；生产 `SGDRegressor` 支持多损失、正则（`alpha`）、`learning_rate` 多种调度、`early_stopping`。
- 教学 每 epoch 都保存所有 loss（内存 O(epochs)）；生产一般只存必要量或快照。
- 教学 用 `np.random.default_rng(seed)` 洗牌；生产用专门 Mersenne/`RandomState` 并允许多 worker 并行打乱。
- 教学 线性模型只有一阶梯度；生产把优化器（`sklearn.utils.optimize` / `SGD`）耦合成通用 classes。

这些就是"教学优先"取舍——优化器与模型的分工语义正确是核心，工程强度和算法鲁棒交给正式 sklearn。

## 练习题

1. 把 `GDRegressor` 的 `learning_rate` 从 `0.05` 改成 `5.0`，观察 `loss_history_` 是否会爆炸（`loss` 越走越大）？为什么？
2. 给全批 GD 调个 闭合 `np.linalg.solve(XaᵀXa, Xaᵀy)` 的闭式答案，对比它们解析到的 `coef_`、`intercept_`、`R²` 是否一致。
3. 手动实现 `lr/(1+0.01*iter)` 调度后再用固定 `lr`，并打印两条 `loss_history_` 曲线，说明调度线更平稳地逼近收敛。
4. `GDRegressor` / `SGDRegressor` 都带 `fit_intercept=True` 时，把 `X` 拼一列 1 后 GD 和 SGD 的 `w_aug[0]` 对应 `intercept_`。若 `fit_intercept=False`，两者 `intercept_` 应恒为 `0.0`，推断为什么。

## 下一章

现在我们有两条路找同一组线性参数：闭式（s06）与迭代优化（s07）。下一步 s08 会把这些归入 `base.EstimatorImpl` 协议、并介绍 `BaseEstimator.set_params` 如何在两个优化器上统一配置 `learning_rate`/`max_iter`；再往后 s09 才用真正的 `sklearn.linear_model` 做 parity 对比。