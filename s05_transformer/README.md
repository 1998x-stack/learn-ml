[返回根目录](../README.md)

# s05：Transformer——学习数据状态，但不预测目标


> 本章口号：**Transformer 学习数据状态，但不预测目标。**
>
> 对应 sklearn 模块: `sklearn.preprocessing`, `sklearn.impute`

## 1. 上一章留下的问题

s04 我们建立了评估闭环：`切分 → 训练 → 预测 → 指标`。但训练之前，原始数据的"形态"往往是杂乱的——量纲不同、分布偏移、甚至有缺失值。

看两列特征：身高（厘米，≈170）和体重（千克，≈65）。它们的均值 / 方差不在同一量级，梯度下降、距离计算等算法会被数值大的特征主导。s05 引入**Transformer**：在喂给模型之前，先改写数据的"状态"——标准化、缩放、填缺失——让算法看到更规整的输入。

Transformer 与 s01–s04 的模型有一个根本区别：**它不预测目标**。

## 2. 本章目标

读完本章，你应该能回答：

- Transformer 和 s01–s03 的 Estimator（分类 / 回归）本质区别是什么？
- `StandardScaler().fit(X_train).transform(X_test)` 与 `StandardScaler().fit_transform(X_test)` 哪里不同？前者为什么才不会泄露？
- 为什么 `scale_`、`mean_`、`statistics_` 末尾带下划线且不出现在 `get_params()`？
- `transform` 为什么必须**复用**训练时学到的统计量，而不是对传入数据重算？

核心思想：**Transformer 学习"数据的状态"，但不预测"目标的值"。**

## 3. 数学直觉

Transformer 的直觉是把"数据长什么样"当成一段备忘，而不是一个判决。

```text
fit(X_train)      -> 记住训练集的 mean_/scale_/min_/max_/statistics_（备忘录）
transform(X)      -> 拿出备忘录，把任意 X（训练或测试）改写成规整形态
fit_transform(X)  -> fit 完立刻 transform（仅限训练集；用在测试集=泄露）
```

"状态"是对**训练集**学一次的：`mean_` 是训练集的均值，`scale_` 是训练集的标准差。测试集合一旦照这套统计去改写，就成了**同一把尺子量训练与测试**——形状真实，这才可比较。

## 4. 数学定义

本章新增三个构件。

```text
StandardScaler   transform: (X - mean_) / scale_      # 每个特征 0 均值 1 方差
MinMaxScaler     transform: (X - min_) / scale_ * (max - min) + min   # 缩到[0,1]
SimpleImputer    transform: X[missing] <- statistics_（列均值 / 中位数）

mean_  = mean(X_train, axis=0)
scale_ = std(X_train, axis=0)                        # StandardScaler
scale_ = max(X_train,0) - min(X_train,0)             # MinMaxScaler（零区间 -> 1.0）
statistics_ = nanmean(X_train, axis=0)               # SimpleImputer.strategy="mean"
```

三个都是"学一个向量 → 改写输入"。唯一不同是学什么、怎么写。

## 5. 最小实现

最朴素的标准缩放就是除均值除标准差——一旦能向量化，三行搞定：

```python
class StandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.scale_   # 复用，不重算
```

那为什么不能 `fit_transform(X_test)`？因为那样 `mean_` 就会变成 **测试集自己的均值**——测试集的分布被抹除，评估就失真了。

## 6. NumPy 向量化实现

三种变换都是一条向量化表达式，零 for 循环：

```python
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def minmax(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def impute(X, stats):
    missing = np.isnan(X)
    X = X.copy()
    X[missing] = np.take(stats, np.where(missing)[1])   # 按列位置取填充值
    return X
```

MinMaxScaler 有个边界坑：若某列训练集上"零区间"（全同值），`max-min == 0` 会除零。教学实现用 `np.where(scale == 0, 1.0, scale)` 把该列的回退成 1.0，让该列保持原值。

## 7. scikit-learn 风格封装

框架轨道把它放进 `src/mini_sklearn/preprocessing/`：

```python
from src.mini_sklearn.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer

sc = StandardScaler().fit(X_train)          # 只学一次训练统计
Xtr_t = sc.transform(X_train)
Xte_t = sc.transform(X_test)                # 用同一套统计改写测试集

imp = SimpleImputer(strategy="mean").fit(X_train)
X_filled = imp.transform(X_train)           # 把 NaN 填成列均值
```

框架轨道遵循统一的 Estimator 协议：`fit` 返回 `self`、学习到的属性带 `_` 结尾、`get_params` 只返回超参数（`with_mean`/`with_std`/`strategy`）、未 `fit` 就 `transform` 报 `ValueError`。`fit_transform` 由 `TransformerMixin` 提供。教学轨道把 `fit_transform` 内联在 `code.py` 自包含，行为一致。

## 8. 相对上一章的变化

上一章（s04）评估闭环里，模型直接吃原始特征。本章在环前面加了一步"状态归一"：

- s01–s04：`X → fit → predict/score`，原始特征直接进入模型。
- s05：`X → fit(Transformer) → transform(X) → 进入模型`，并且 `transform` 永远用训练集的统计量。
- 新增三种 Transformer：中心化（StandardScaler）、量纲压缩（MinMaxScaler）、缺失填补（SimpleImputer）。

一个反复出现的反模式就是 `StandardScaler().fit_transform(X_test)`——它在测试集上重算统计量，抹除了测试集本身的分布形状。s05 的目标之一就是把这条反模式钉死在教材里。

## 9. 运行实验

```bash
# 教学轨道：自包含可运行（反模式 + 正确模式 + 缺失填充）
python s05_transformer/code.py

# 框架轨道：pytest 验证 numerical + api
python -m pytest tests/numerical/test_preprocessing.py tests/api/test_api_preprocessing.py -q
```

教学轨道示例输出（种子 `np.random.default_rng(42)`）：

```text
=== 反模式：对测试集重复 fit_transform（数据泄露）===
测试集被自己的统计量标准化 -> 测试集均值为 [-0.  0. -0.]，标准差 [1. 1. 1.]，物理含义被抹除

=== 正确模式：fit 在训练集，transform 复用统计量 ===
训练集缩放后均值   [ 0. -0.  0.]，std [1. 1. 1.]
测试集缩放后均值   [-0.3215 -0.0108 -0.3672]，std [1.301  1.2756 1.3556]

[PASS] s05: Transformer 学习数据状态但不预测目标；transform 复用训练统计而非重算
```

测试预期输出：

```text
17 passed in 0.10s
```

## 10. 与 sklearn 对比

框架轨道是 `sklearn.preprocessing` / `sklearn.impute` 两个模块的缩小版：

```python
# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
# 本章
from src.mini_sklearn.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer
```

关键差异（我们刻意简化）：

- sklearn 的 `MinMaxScaler` 还有 `feature_range` 之外的 `clip`、`copy` 等；本章只留 `with_min`/`with_max`。
- sklearn 的 `SimpleImputer` 支持 `most_frequent`/`constant`/`add_indicator` 和轴向选择；本章只留 `mean`/`median`。
- sklearn 的 `StandardScaler` 还有 `copy`、`unit_variance`（新版本）、有稀疏矩阵支持；本章只针对稠密二维数组。

## 11. 失败案例和边界条件

- **对测试集重复 `fit_transform`**：测试集统计量被自己改写，分布形状被抹除（均值归零）——数据泄露。
- **MinMaxScaler 零区间列**：某列全同值导致 `max-min = 0` 除零；用 `np.where(scale==0, 1.0, scale)` 回退。
- **未 fit 就 transform**：`mean_`/`scale_`/`statistics_` 不存在，应该抛 `ValueError`（复用 `check_is_fitted`）。
- **特征数不匹配**：`transform(X)` 的列数 ≠ `fit` 时的 `n_features_in_`，应报明确错误。
- **缺失值**：`transform` 时把该列的 `statistics_` 填进缺失位置；全部缺失的列算不出均值，留给后续章节处理。

## 12. 教学实现与生产实现的差距

- 教学的 `transform` 总是返回二元 `ndarray`；sklearn 保留原 dtype 与稀疏结构，并支持 `copy` 参数。
- 教学 `fit` 直接读列统计；sklearn 统一走 `_check_feature_names`、`n_features_in_`、`allow_nan` 的完整输入校验。
- 教学 `SimpleImputer` 只处理 `nan`；sklearn 允许自定义 `missing_values`（含 `pd.NA`）与 `add_indicator`。
- 教学 `StandardScaler` 的 `scale_` 恒为列 std；sklearn 在 `unit_variance=True` 时按 `ddof=0` 但另有选项。

这些仍是"教学优先"的取舍——标准化 / 归一化 / 缺失填充的语义正确是核心，工程选项留给正式 sklearn。

## 13. 练习题

1. 把 `StandardScaler().transform(X_test)` 改成 `StandardScaler().fit_transform(X_test)`，观察 `X_test` 均值被抹除的现象，再解释为什么两者结果不同。
2. 给 `MinMaxScaler` 的一列构造恒为常数的输入，验证 `scale_` 被回退成 `1.0` 且变换后该列保持原值。
3. 在 `fit` 前直接调用 `transform`，断言抛 `ValueError`；再用 `check_is_fitted(self, ["statistics_"])` 检查错误信息。
4. 给 `SimpleImputer` 构造一列全为 `NaN`，观察 `statistics_` 变成 `nan`，再讨论这是否可接受。

## 14. 下一章

现在我们能：把杂乱数据规整成规整的"状态"（标准化 / 归一化 / 缺失填充），且不泄露测试集统计量。

下一章进入 `s06`：用这些规整的数据学**线性回归**——"最小二乘 / 梯度下降"的参数估计，让 Transformer 规整好的输入真正产生可解释的系数。