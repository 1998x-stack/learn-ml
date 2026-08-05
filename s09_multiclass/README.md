# s09：One-vs-Rest 多分类——不修改二分类器，用包装器扩展能力

[返回根目录](../README.md)

> 本章口号：**不修改基础二分类器，用包装器扩展能力。**
>
> 对应 sklearn 模块: `sklearn.multiclass, sklearn.preprocessing`

## 1. 上一章留下的问题

s08 把线性模型带到了**二元分类**：`LogisticRegression` 只会答「是 / 不是」。可现实中大多是**多分类**——手写数字 0–9、鸢尾花的三个品种、新闻的十个板块。s08 结尾明说「多类留给 s09」。

s09 的答案和「重写一个 softmax 模型」南辕北辙：它**不碰**已经写好的二分类器，而是用一层**包装器**把 s08 的 `LogisticRegression` 装成多分类。武器是一条朴素思想——**one-vs-rest（一对多，OvR）**：

```text
要判 k 类？训练 k 个「是不是第 c 类」的二分类器，
每个分类器只负责回答一个 yes/no：「你是三角形吗」「你是正方形吗」「你是圆形吗」。
拿到样本，问遍所有分类器，谁最有把握就把样本判给谁。
```

单模型是 `Learner`，这里引入的是**元估计器（meta-estimator）**——一个「训练模型」的模型，它组合基础学习器而非发明新算法。

## 2. 本章目标

读完本章，你应该能回答：

- 为什么要 OvR 而不是直接重写一个多分类模型？（复用、解耦、基础模型无感知）
- 元估计器与普通估计器差在哪？`fit` 在 OvR 内部到底循环了几次？
- 为什么 OvR 用 `clone` 每类重建一个模型，而不是反复复用同一个已拟合模型？
- OvR 的 `predict` 为什么是「正类概率矩阵 → argmax」？
- OvR 与 OvO（one-vs-one）各自偏好什么时候？
- `LabelEncoder` 怎么把 `{triangle, square, circle}` 压成 `0..k-1`？

核心思想：**不改模型，只改组合方式——用 k 个二分类器拼接出多分类。**

## 数学直觉

多分类的判定可以看成 k 个「独立投票员」。第 `c` 个投票员只被问一个问题：`“这个样本是类 c 吗？”`。它内部用 s08 的 sigmoid 把线性得分压成概率：

```text
第 c 个模型   y_c,i = 1  if  y_i == c   else   0      (one-vs-rest 目标)
        p_c(x) = sigmoid(x @ coef_c + intercept_c)     (「是第 c 类」的可能)
最终预测       预测 = classes_[ argmax_c p_c(x) ]       (谁最自信就选谁)
```

于是「多分类」退化成「k 个独立的二分类」，每个二分类都已经是 s08 验证过的熟脸。这就是元估计器的威力：**不改基础分类器的一行代码，能力被扩到任意类别数。**

## 数学定义

```text
类目标签    classes_ = sorted(unique(y))                        # 排序后的标签
编码        enc_i = LabelEncoder(y)_i                            # 0..k-1
OvR 拟合    对每个 c in classes_：
                y_c,i  = (y_i  == c)                             # 一对一 vs 其余
                fit  binary_estimator_c  on  (X, y_c)            # 每个都独立训练
得分矩阵    S_{i,c} = predict_proba_c(X_i)[:, 1]                 # 正类概率，(n,k)
预测        yhat_i  = classes_[ argmax_c S_{i,c} ]               # 最大置信度的类
概率(可选)   P_{i,c} = S_{i,c} / Σ_c S_{i,c}                     # 行归一化成概率
```

关键在第 5 行：`predict_proba[:, 1]` 取的是「**正类**」那一列——对第 `c` 个模型，它就是「是类 c 而非其余」的置信度。把这 k 个置信度并列成 (n,k) 矩阵，`argmax` 沿类轴取整，`classes_[位置]` 映射回原始标签。

## 最小实现

叠加在 s08 的 `sigmoid` 之上，最核心的包装只有一段循环 + 一行决策：

```python
class OneVsRestClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimators_ = [self._clone(self.estimator) for _ in self.classes_]
        for c, est in zip(self.classes_, self.estimators_):
            y_bin = (y == c).astype(float)            # "是 c " vs " 其余"
            est.fit(X, y_bin)

    def predict(self, X):
        S = np.column_stack([e.predict_proba(X)[:, 1] for e in self.estimators_])
        return self.classes_[np.argmax(S, axis=1)]
```

`_clone` 用相同超参数重建一个**未训练**的实例——这正是框架里 `clone()` 的职责。s02 我们为它铺过路，这里派上用场。

## NumPy 向量化实现

`predict` 里的得分矩阵一行 `column_stack` 就拼好（k 个 `@` 各算一次线性得分）：

```python
def _scores(self, X):
    return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])

def predict(self, X):
    best = np.argmax(self._scores(X), axis=1)          # (n,)，同分取低类号
    return self.classes_[best]
```

和 s08 一样，`predict` 依旧是「概率的 argmax」——只不过这次概率来自 k 个模型。`np.argmax` 在同分时返回**较小的索引**，恰好实现了「平分时让给序号更小的类」这个约定。

## scikit-learn 风格封装

框架轨道新增 `src/mini_sklearn/multiclass/`，提供 `LabelEncoder` 与 `OneVsRestClassifier(estimator)` ——后者是框架首个真·元估计器：

```python
from src.mini_sklearn.linear_model import LogisticRegression
from src.mini_sklearn.multiclass import LabelEncoder, OneVsRestClassifier

enc  = LabelEncoder().fit(y)                # y 任意标签 → classes_ 排序
y_le = enc.transform(y)                      # 0..k-1 的密集整数
clf = OneVsRestClassifier(LogisticRegression()).fit(X, y)
print(clf.estimators_)                      # k 个独立克隆，各拟合 (X, y=='c')
print(clf.predict(X))                       # classes_[argmax]，(n,)
print(clf.predict_proba(X))                # 每类置信度，行归一化后和=1
```

- `OneVsRestClassifier(estimator)`：`fit` 把 `estimator` 克隆 k 次，第 i 个克隆拟合 `(X, y == classes_[i])`。
- `predict` 用各克隆的**正类概率**（`predict_proba[:,1]`）拼成 (n,k) 得分，`argmax` 取最大置信度。
- `get_params` 返回 `{"estimator": <原实例>}`：为不与框架 `clone`（`__class__(**get_params())`）冲突，教学上不铺开 `estimator__*` 嵌套参数。
- `predict_proba` 把得分矩阵按行归一化，方便对比各类的相对置信度（非严格概率，教学够用）。
- 编码/解码由 `LabelEncoder` 包起：`fit` 记 `classes_`，`transform`→`0..k-1`，`inverse_transform` 还原。

## 相对上一章的变化

| | s08（二元分类） | s09（多分类 OvR） |
|---|-----|-----|
| 适用的二分类器 | 1 个 `LogisticRegression` | k 个 `LogisticRegression` 克隆 |
| 新机制 | sigmoid + 交叉熵 | 元估计器 + `clone` + `LabelEncoder` |
| 决策 | `argmax([1-p, p])` | `argmax(正类概率列)`，列来自不同模型 |
| 训练 | 拟合一次 | `fit` 内部循环 k 次，各拟一个子问题 |
| 标签 | `classes_` 两份 | `classes_` k 份 + 密集编码 `0..k-1` |
| 职责 | 一个学习算法 | 组合学习算法的包装器 |

`s09` 新增了 `tests/numerical/test_multiclass.py`（三丘可分离、恰好 3 类、准确率高、编码回环）与 `tests/api/test_api_multiclass.py`（fit→self、未 fit 就报错、参数面、`clone` 独立性）。

## 运行实验

```bash
# 教学轨道：自包含可运行，零 import src 手搓 OvR + LabelEncoder
python s09_multiclass/code.py

# 框架轨道：pytest 验证多分类策略 + 编码
python -m pytest tests/numerical/test_multiclass.py tests/api/test_api_multiclass.py -q
```

教学轨道示例输出（三个相距很远的 blob）：

```text
=== 元估计器：每个克隆拟合 (X, y=='c') 的二元问题 ===
estimators_ 数量    = 3（= 类别数 3）

=== argmax 决策 ===
得分矩阵形状        = (180, 3)（180 个样本 × 3 类）
行和 ≈ 1            = True
  y=0  predict=0  proba=[0.982 0.018 0.   ]
  y=1  predict=1  proba=[0.002 0.998 0.   ]

[PASS] s09: 不改一只用 k 个二分类器的包装，得到多分类，OvR 训练准确率 1.000
```

强分离三丘上 OvR 各候选模型都极有把握，argmax 几乎总赢。

## 与 sklearn 对比

```python
# sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
# 本章
from src.mini_sklearn.multiclass import OneVsRestClassifier, LabelEncoder
```

- sklearn 的 `OneVsRestClassifier` 也包装任意 `estimator`，语义一致；它的 `decision_function` 用原始得分比较，sklearn 在有线性二分类器时走这条路，概念上同构。
- sklearn 当然也提供 `predict_proba`，还支持多标签、`n_jobs` 并行、类不平衡等；本章只取最朴素的一种。
- sklearn 的 `LabelEncoder` 的 `transform` 会接受并查错；本章同样报「not seen during fit」。
- sklearn 原生 `LogisticRegression` 也有 `multi_class`（原 one-vs-rest 版本）与真正的 softmax；本章用包装器显式演示 OvR，而 softmax 一词留给 s09 习题与后续。

## 失败案例和边界条件

- **类别数 < 2**：只有一个类时划分「一对一 vs 其余」没有意义，OvR应抛 `ValueError`而非默默返回全一类。
- **标签不是整数**：OvR 内部要 `y == c` 去构造二分类；`LabelEncoder` 先把任意标签（字符串、乱序整数）编码成 `0..k-1`，解码时 `inverse_transform` 还原。
- **未 fit 就预测**：`predict`/`predict_proba` 都先 `check_is_fitted` → `ValueError`，api 测试锁定。
- **特征列不匹配**：`predict` 时 `X` 列数与 `n_features_in_` 不齐要抛错（沿用框架校验）。
- **平局**：`np.argmax` 在同分时取**第一个最大类**（较低类 id）——确定性，且不随机。
- **缺失的概率界面**：OvR 依赖基础分类器有 `predict_proba`；若基础模型没有该接口，则需按其自身输出约定兜底（本章的 `LogisticRegression` 有 `predict_proba`，所以直接安全）。
- **类别不平衡的代价**：OvR 每个子问题 `正例 n_c` 面对其余 `(N - n_c)` 个反例；当类数 k 很大时正例占比仅 `1/k` 左右，梯度被反例主导、少数类易欠拟合——这正是它与直接 softmax 单模型最大的差距。教学用「三等份、互相远离」的合成数据规避这条，聚焦 OvR 最干净的面貌。

## 教学实现与生产实现的差距

- 教学 `OneVsRestClassifier` 手工克隆 `estimator`，假设基础模型是 `LogisticRegression`；生产 `clone` 通用任意 estimator 并从 `estimator__param` 超参数链路做嵌套配置。
- 教学把 `predict_proba` 简化为「得分矩阵行归一化」；生产在训练期用多元逻辑回归/或 Platt，逐步校准概率。
- 教学不处理类别不平衡、权重、核化基础模型；生产有 `class_weight` / `n_jobs` / 并行 `fit`。
- 教学展示 OvR 朴素转移；生产还有 OvO（两两比较）作为另一策略，下一章可能有选择。

## 练习题

1. 把 `argmax(S, axis=1)` 换成 `axis=0`，预测结果会怎样崩掉？为什么「沿样本轴取最自信类」才是正确的？
2. 给 `OneVsRestClassifier` 传入一个「只有 `decision_function`、没有 `predict_proba`」的桩分类器，`fit`/`predict` 分别在哪个方法挂掉？若改用 `decision_function` 当得分，`predict` 会不会也崩？理解 OvR 到底依赖哪个接口。
3. 把 `enc.transform(y)` 去掉，直接把字符串型 `y` 送进 `OvR`，在 `y == c` 处会发生什么？为什么 `LabelEncoder` 是不可省的一环？
4. 手动把 `three_class_data` 的某个 blob 的 `per` 改成 `200`，让类严重失衡，`acc` 会跌到多少？OvR 在不平衡下为什么「大类永远赢」？
5. 实现一个对照的 OvO（one-vs-one：每对类一个分类器，共 `k(k-1)/2` 个），并比较训练时长和准确率。什么时候值得多花这笔钱？

## 下一章

s09 用「包装器」把二叉分类器扩展成多类，但这种包装仍要求基础模型是可训练的。下一章 s10 将把这些估计器和 预处理/数据 收进一个统一的 `Pipeline`，让「特征工程 + 模型」整条链可一次 `fit`、一次 `predict`，也为 s14–s16 的交叉验证与 Grid Search 铺路。
