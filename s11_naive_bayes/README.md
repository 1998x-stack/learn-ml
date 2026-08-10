[返回根目录](../README.md)

# s11：Naive Bayes——从距离判断转向概率推断


> 本章口号：**强假设可以换来简单、快速和稳定。**
>
> 对应 sklearn 模块: `sklearn.naive_bayes`

## 1. 上一章留下的问题

s10 的 KNN 靠「距离」判断一切：算 query 到每个邻居有多远、让最近的 k 个投票。
但「距离」这种实值比较有两个坏毛病：一是**贵**——每次 predict 都要把 query 和
全部训练点比一遍；二是**不稳**——特征一多、类别一多，离谁近离谁远很容易糊掉。
而且 KNN 只给答案、不给「这个答案有多把握」。

s11 换成**概率推断（generative）**：不再问「谁离得近」，而是问「每种类别有多大
**概率**生成了这个样本」。这从一个纯 `argmin(距离)` 的判别式思路，转向真正把类别
当成「生成器」、用贝叶斯规则做后验选择——这是第一种**生成式（generative）分类器**。

## 2. 本章目标

读完本章，你应该能回答：

- 贝叶斯公式 `P(y|x) ∝ P(y) × P(x|y)` 里，先验与似然各是什么、怎么估计？
- 为什么说 NB 是「朴素」的？「给定类别后特征相互独立」这个假设会牺牲什么、换来什么？
- 为什么要全程用**对数空间**，而不是直接连乘那些很小的概率？（连锁下溢）
- 高斯朴素贝叶斯：为什么 `P(x_d|y)` 用**每类别、每特征**的高斯密度来建模？
- `predict_proba` 怎么从 `joint_log_likelihood` 一步 softmax 出来？行为什么和为 1？
- `var_smoothing`（方差平滑）在零方差特征上的防护作用是什么？

核心思想：**`P(y | x) ∝ P(y) × ∏_d P(x_d | y)`，而且全部在 log 空间算。**

## 数学直觉

上一章 KNN `nearby → vote`；本章改成 `generator → Bayes`：

```text
判别/非参数（s10）:   distance(query, neighbors) → majority vote
生成式（s11）  :   P(x|y) · P(y)             → max posterior class
```

对每个样本 `x`，我们给每种类别 `k` 打一个「后验分」：

```text
P(y=k | x)  ∝  P(y=k)        ×  P(x | y=k)
              ^先验(每类比例)    ^似然(该类生成出 x 的概率)
```

「朴素」指的是我们把最棘手的联合分布 `P(x|y)` 拆成每个特征独立的条件概率连乘：

```text
P(x | y) ≈ ∏_d P(x_d | y)        ←  特征在给定类别后相互独立（强假设）
```

这个假设在数据上几乎从不成立——两个特征常常相关——但**它居然很能用**：假设越强，
要估的参数越少（每类只需每特征一组参数），模型就越简单、训练越快、越抗过拟合。
这就是口号：**强假设可以换来简单、快速和稳定。**

## 数学定义

高斯朴素贝叶斯假设每类别、每特征都服从一维正态分布，其参数用训练数据估计：

```text
θ[d,k]   = mean_d(X belonging to class k)         特征 d 在类 k 的均值
σ²[d,k]  = Var_d(X | class k) + var_smoothing     加上一个极小平滑防零方差
log π_k  = log( n_k / N )                          类 k 的 log 先验

每样本、每类的后验打分（对数空间）：
  log P(y=k | x) =  log π_k  +  Σ_{d=1}^D log N(x_d ; θ[d,k], σ²[d,k])

  N(x; μ, σ²) = 1/sqrt(2πσ²) · exp( -(x-μ)² / (2σ²) )
  log N(...) = -½ log(2πσ²) - (x-μ)² / (2σ²)
```

`predict_proba` 对上面 (n,K) 的 log 打分做**行内 softmax**（每行减最大值再 exp、归一化），
得到 (0,1) 且每行和为 1 的类别概率；`predict` = `argmax_k` 选分数最高的类别。

## 5. 最小实现

```python
class GaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # 每类、每特征：均值 + 方差（附平滑）
        for c in self.classes_:
            Xc = X[y == c]
            theta.append(Xc.mean(axis=0))
            sigma.append(Xc.var(axis=0) + var_smoothing)
        # log 先验
        self.class_log_prior_ = np.log(count / N)
        return self

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)          # (n, K) 对数后验打分
        exp = np.exp(jll - jll.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
```

一只 `gaussian_log_pdf` 把整段的精髓都写完了：

```python
def gaussian_log_pdf(x, mu, var):
    return -0.5*np.log(2*np.pi*var) - 0.5*(x-mu)**2 / var
```

**千万不能**写成 `pdf = 1/sqrt(2πσ²)*exp(...)` 再连乘：N 个小概率乘起来会**下溢成 0**，
softmax 后就全 NaN。对数空间把「乘」变「加」，连乘永不塌缩——这就是为什么本章死守 log。

## NumPy 向量化实现

框架版把「每类每特征的 log 密度」整张算出来一次广播乘出来（一次性 (n, D, K)）：

```python
log_pdf = (
    -0.5 * np.log(2 * np.pi * self.sigma_)[None, :, :]
    - 0.5 * (X[:, :, None] - self.theta_[None, :, :])**2 / self.sigma_[None, :, :]
)                      # (n, D, K)
jll = log_pdf.sum(axis=1) + self.class_log_prior_[None, :]   # 各特征 log 相加
P   = softmax(jll)                                            # → (n, K)
```

`X[:, :, None]` 与 `theta_[None, :, :]` 广播成 `(n, D, K)`，把「n 个样本 × D 个特征 ×
K 个类别」的高斯对数密度一次算完，再沿特征轴求和——这才是 NumPy 一行的向量化威力。

## scikit-learn 风格封装

框架轨道新增 `src/mini_sklearn/naive_bayes/`（`GaussianNB` / `MultinomialNB` /
`BernoulliNB`）：

```python
from src.mini_sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, y)                    # theta_ / sigma_ / class_log_prior_ / classes_
print(clf.predict_proba(X))      # (n,K)，行和=1，∈(0,1)
print(clf.predict(X))            # classes_[argmax] → 类别标签
```

- 三个类都满足 Estimator 协议：`fit`→`self`、`classes_`、`n_features_in_`、
  predict 前必须有 fit（`check_is_fitted` 否则 `ValueError`）。
- `get_params` 只暴露超参：`GaussianNB(var_smoothing=…)`，`MultinomialNB(alpha)`/
  `BernoulliNB(alpha)`——学到的状态（`theta_`/`sigma_`/`feature_count_`…）永不进参数表。
- `MultinomialNB` 期待非负计数（如词频）、`BernoulliNB` 期待 0/1 二值特征，都支持
  Laplace 平滑 `alpha`。

## 相对上一章的变化

| | s10（KNN） | s11（Naive Bayes） |
|---|-----|-----|
| 思想 | 距离 + 邻居投票（判别式） | 先验 × 似然（生成式） |
| fit 做了什么 | 只存档训练数据 | 学每类**参数坐标**（θ、σ、π） |
| 判断依据 | 距离远近 | 后验概率 `P(y|x) ∝ P(y)·P(x|y)` |
| predict 复杂度 | 逐点搜索，推理最重 | 一条公式扫一遍，推理极轻 |
| 输出 | 类别 / 均值 | 每类概率 `predict_proba`（行和 1） |
| 稳健性 | 怕维度、怕特征尺度 | 强假设换来简单、快速、稳定 |

s10 KNN「训练轻、推理重」；s11 正相反：训练记一小撮统计量，predict 只是把 log 打一遍
分式子扫干净——推理（几乎）=免费。这正合口号**「强假设可以换来简单、快速和稳定」**。

## 运行实验

```bash
# 教学轨道：零 import src，手搓 log 空间 GaussianNB
python s11_naive_bayes/code.py

# 框架轨道：pytest 验证数论正确性 + API 协议
python -m pytest tests/numerical/test_naive_bayes.py tests/api/test_api_nb.py -q
```

输出示例（两个中心很近的二维簇，靠概率才分得开）：

```text
=== 高斯对数空间朴素贝叶斯 ===
训练数据形状          = (120, 2)
学到的每类均值 theta_ =
  类别 0: [1.   1.07]
  类别 1: [2.09 2.21]
训练集 accuracy      = 0.95
predict_proba 行和   = [1. 1. 1.] （应为 1）

[PASS] s11: 对数空间高斯朴素贝叶斯成立——用「先验 × 似然」的贝叶斯推断还原近距离二维簇
```

## 与 sklearn 对比

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
```

- 三者的类名、`var_smoothing` / `alpha` 默认值、以及 `fit/predict/predict_proba` 签名
  逐一对齐。sklearn 的 `GaussianNB` 也会存 `theta_`/`sigma_`/`class_prior_`/
  `class_count_`/`class_log_prior_`——只是它 `theta_` 形状是 `(n_classes, n_features)`，
  本章为教学把矩阵记成 `(n_features, n_classes)`，语义完全一致。
- `predict_proba` 等价于对 `joint_log_likelihood` 做 `from_logits=True` 的 softmax；
  sklearn 内部同样稳定（减去行最大后再指数）。
- `MultinomialNB` 和 `BernoulliNB` 实现了相同的 Laplace 平滑 (`alpha`)，区别仅在
  连续（高斯）、计数（多项）、0/1（伯努利）三种特征假设。

## 失败案例和边界条件

- **未 fit 就 predict**：`check_is_fitted` 拦截，抛 `ValueError`（如每个 API 测试锁定）。
- **特征列不匹配**：predict 的 `X` 列数与 `n_features_in_` 不齐 → `ValueError`。
- **零方差特征**：训练类别内某特征恒定 → var=0，直接除零；`var_smoothing`（默认 1e-9）
  给每个方差垫一个小数、绝不除零。
- **连乘下溢**：特征多时 `∏P(x_d|y) → 0`；全程 `log` 相加即可杜绝（这是本章的核）。
- **独立性被违背**：特征紧密相关时朴素假设失真，概率可能偏门——但那也常「够用」。

## 教学实现与生产实现的差异

- 教学用暴力 `theta_`/`sigma_` 密集数组、按特征 `log` 相加；生产端对高维数据用
  稀疏矩阵、并缓存 `joint_log_likelihood` 供多次 `predict` 复用。
- 教学只实现 `Gaussian` / `Multinomial` / `Bernoulli` 三种似然；生产还支持 `CategoricalNB`、
  `ComplementNB` 等家族。
- 生产端 `GaussianNB` 的 `theta_` 是 `(n_classes, n_features)`，且对 `alpha` 提供
  `fit_prior` 开关（用不用经验先验）；本章固定使用经验先验、教学化形状。

## 练习题

1. 用 3 行实现 `predict_proba`：对 `joint_log_likelihood(jll)` 做「减行最大 → exp → 归一化」，
   验证结果与直接 `softmax(jll)` 完全一样。
2. 给 `GaussianNB` 的 `predict` 传一个 `var_smoothing` 为 0 的模型，看零方差类会发生什么；
   再传一个合理的值恢复稳定。
3. 把 `sigma_` 从「高斯方差」换成「全类共享同一方差」（同方差朴素贝），零方差问题是否消失？
4. 用 `MultinomialNB` 在词频矩阵（如 Gutenberg 语料的 word-counts）上做文本分类，比较它对
   无效词频（0 词）的平滑行为与 `BernoulliNB`。
5. 思考：为什么「给定类别后独立」能「换来简单、快速和稳定」？反过来，当特征强相关时你
   会预期朴素贝在哪些数据上崩？

## 下一章

贝叶斯把判断从「距离」换成「概率」，却仍需在全部特征上一遍扫过。下一章 **s12 决策树**
走向完全不同的路径——**用一组递归的 if-else 规则**去切分特征空间，不词太概率、直接长
成一颗可解释的树，天然处理非线性与特征交互。