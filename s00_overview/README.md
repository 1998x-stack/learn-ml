# s00：全景导览（Panorama）

[返回根目录](../README.md)

> 本章无代码，只搭建整个 mini-sklearn 课程的心智地图。
>
> 对应：`versions.md` 的总体方案、不变核心、两条演进主线、六阶段路线、双轨结构与版本树。

## 上一章

无。这是全系列的第 0 站，用于建立整条学习路线的全景。

## 本章目标

读完本章，你应该在心里建立一张「MiniSklearn Core」的地图。整个课程始终围绕一个不变的最小内核实心展开：

```text
MiniSklearn Core
    =
Estimator Protocol
    + Data Contract
    + Fitted State
```

同时理解两条并行演进的主线（算法演进线 + 框架演进线），以及贯穿全书的教学/框架双轨结构。本章不写任何代码，只为 s01–s04 的「最小监督学习闭环」做概念铺垫。

## 版本树导览

`versions.md`（§一三 版本与章节映射）把全部章节装进一条版本树。本迭代只覆盖 **v0.1（s01～s04，最小监督学习闭环）**，其余版本作为后续迭代的路线图呈现。

| 发布版本 | 章节 | 目标 |
| ------ | ------- | ------------------ |
| `v0.1` | s01～s04 | 最小监督学习闭环 |
| `v0.2` | s05～s09 | 预处理和线性学习 |
| `v0.3` | s10～s13 | 经典非线性模型 |
| `v0.4` | s14～s16 | 交叉验证、搜索、Pipeline |
| `v0.5` | s17～s18 | 无监督学习 |
| `v0.9` | s19 | 第三方 Estimator 与兼容性 |
| `v1.0` | s20 | 完整 mini-sklearn |

## 稳定内核（The Minimal Core）

`versions.md`（§一）反复强调：迭代式课程的诀窍是「找到一个始终不变的内核，每一章只增加一个机制，最后再把全部机制组合成完整系统」。

机器学习框架的不变核心并非某个梯度下降循环——因为决策树、KNN、PCA、线性回归的训练过程各不相同。真正稳定的是 **Estimator 生命周期协议**：

```python
model = Estimator(**hyperparameters)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

完整生命周期为：

```text
初始化超参数
    ↓
验证输入数据
    ↓
学习模型参数
    ↓
保存 fitted attributes
    ↓
检查模型是否已训练
    ↓
预测或变换新数据
    ↓
评估结果
```

用 `MeanRegressor` 表达这套共识的伪代码如下（取自 `versions.md` §一）：

```python
class MeanRegressor:
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.constant_ = y.mean()
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        check_is_fitted(self, "constant_")
        X = check_array(X)

        return np.full(X.shape[0], self.constant_)
```

关键点：**每个后续模型都不改变调用方式**。

```python
LinearRegression().fit(X, y)
LogisticRegression().fit(X, y)
DecisionTreeClassifier().fit(X, y)
KMeans().fit(X)
PCA().fit_transform(X)
```

## What is the overall route

`versions.md`（§三）把 20 章主线分成六个阶段，从前到后首尾相接：

```text
阶段一：最小学习闭环（s01-s04）
   ↓
阶段二：线性学习（s05-s09）
   ↓
阶段三：经典非线性模型（s10-s13）
   ↓
阶段四：评估与组合（s14-s16）
   ↓
阶段五：无监督学习（s17-s18）
   ↓
阶段六：扩展与集成（s19-s20）
```

这六阶段回答的是同一个问题：如何从一个单一的 `fit`/`predict` 出发，一步步构建出完整的 mini-sklearn。

## 双轨说明

`versions.md`（§六）指出：直接照搬「每章一份完整框架代码」会产生大量重复；完全共享一个包又会让学习者看不清每一章到底新增了什么。因此采用 **双轨结构**。

**教学轨道**——`s01…s20/code.py`：

- 每章独立运行
- 只保留本章核心
- 代码优先可读
- 可以适度重复
- 不做复杂性能优化
- 各章之间只有概念依赖，不 import 前一章目录

**框架轨道**——`src/mini_sklearn/`：

- 不重复实现公共逻辑
- 使用正式包结构
- 保留类型注解
- 支持完整测试
- 最终由 s20 使用
- 逐步接近工程实现

两条轨道的关系：

```text
教学 code.py
    解释原理、突出机制
           ↓
src/mini_sklearn
    整合、复用、稳定化
```

这也呼应「教学实现不是生产实现」的准则。

## 模块定位

本模块是整套学习路线的 **第 0 站**：它不发散、不写代码，只铺开「最小闭环」的概念起点。后续 v0.1 的 s01–s04（训练状态 → 统一接口 → 数据契约 → 评估闭环）都将从这里的地图和最小内核出发，为本迭代「建立一个最小监督学习闭环」做好铺垫。
