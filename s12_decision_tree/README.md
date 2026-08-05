# s12：决策树——递归划分的 if-else 规则集

[返回根目录](../README.md)

> 本章口号：**一次划分很弱，递归划分形成复杂边界。**
>
> 对应 sklearn 模块: `sklearn.tree`

## 1. 上一章留下的问题

s10 用「距离」、s11 用「概率」给样本定类。两者都是一次性拿全部特征做全局判断：
KNN 每次 predict 要探遍训练点（推理重），NB 假设特征独立（假设强）。它们都给不出
「为什么是这个答案」的**人可读解释**。

s12 换一条完全不同的路：**把判断写成一棵可解释的、递归的 if-else 问句树**。预测一个
样本不再是一次全局打分，而是从根出发、沿着「特征 ≤ 阈值？」一问一答地走到某个叶子，
叶子里的多数类就是答案。这条路天然非线性——每一条从根到叶子的路径都是一组合取规则，
多棵子树拼在一起就能表达任意复杂的决策边界。

## 2. 本章目标

读完本章，你应该能回答：

- 为什么**单次划分（一个 if-else）很弱**？它一条直线只能切一个特征，能画出的边界有限。
- 为什么**递归划分**能形成复杂边界？先在特征上切一刀，再对左右各自递归切——边界是
  一层层"拼"出来的，可组合、可弯折。
- `gini`（基尼）与 `entropy`（信息熵）各怎么算？都在度量「一个节点有多杂」，纯节点=0。
- 怎么找最优划分？对每个特征排序、在相邻**不同值**的中点试刀，取子节点加权不纯度最小的。
- `max_depth` / `min_samples_split` 怎么防过拟合？树越深边界越复杂，越容易记住噪声。
- `feature_importances_` 是什么？每个特征分摊的「整体加权不纯度下降」，归一化求和≈1。

核心思想：**`split` → 将数据沿某特征阈值一分为二 → 对左右递归 `grow`，直到纯/到顶。**

## 3. 数学直觉

上一章的判断是「打分」，本章是「问路」：

```text
判别/参数（s06）：   y = X @ w + b                     一次线性组合
判别/距离（s10）：   query 与邻居的距离               全局最近
概率生成（s11）：   P(x|y)·P(y) → max                 每条特征对 log 打分
递归规则（s12）：   x[feature] ≤ thr → 左/右 → 递归  一问一答走到叶子
```

树的「问路」本质是层级判断：根节点先问一个最有区分度的问题，把数据切成两块；
每一块内部再各自找对自己最有区分度的问题……直到再也切不动（叶子已纯 / 到深度 /
样本太少）。每个 leaf 回答一个「一片区域的多数答案」。

## 4. 数学定义

**不纯度**（impurity）：一个节点越"杂"（各类混在一起），越难预测，值越大。两个经典度量：

```text
gini      = 1 - Σ_k p_k²          分类（v0.3.4）
entropy   = -Σ_k p_k log2(p_k)    分类（v0.3.5）
variance  = mean within-node      回归（v0.3.5，MSE 的等价形式）
```

纯节点的 gini=entropy=0；两类各一半 → gini=0.5、entropy=1.0。

**最优分裂**：对特征 `f`，把样本按该特征值排好序，对相邻的两个不同值取中点
`threshold=(v[i-1]+v[i])/2` 作为候选刀口切出左(L)、右(R)两组，计算**子加权不纯度**：

```text
split_impurity = (n_L / n) · impurity(L) + (n_R / n) · impurity(R)
```

遍历所有特征、所有候选刀口，选 `split_impurity` 最小的那个 `(f, threshold)`。

**递归生长与停止**：建好内部节点后，对左/右子集各自再调用一次 `grow`（深度 +1）。
停止条件（括号即为防过拟合旋钮）：

```text
impurity == 0            → 纯了，停
depth >= max_depth       → 到深度上限，停
n_samples < min_samples_split → 样本太少，停
len(unique(y)) < 2       → 只有一类，停
```

**feature_importances_**：对每个内部节点，计算其**加权不纯度下降量**

```text
(n_node / n_root) · ( impurity_node - (n_L·imp_L + n_R·imp_R)/n_node )
```

分摊给该节点的切分特征；全部累加后归一化到求和≈1。

## 5. 最小实现

```python
@dataclass
class Node:
    value; feature; threshold; left; right; n_samples; impurity

def best_split(X, y, w, criterion):
    for f in range(n_features):                 # 对每个特征
        order = np.argsort(X[:, f])
        xs, ys = X[order, f], y[order]
        for i in range(1, n):                   # 每个相邻不同值的中点
            if xs[i] <= xs[i-1]: continue
            thr = (xs[i-1] + xs[i]) / 2
            imp = gini(y[:i])*len+...  # 加权子不纯度最小者胜
    return best_f, best_thr

def grow(X, y, w, depth, criterion, max_depth, min_samples_split):
    node = Node(value=majority(y), impurity=gini(y))
    if node.impurity == 0 or depth>=max_depth or len(y)<min_samples_split:
        return node                              # 纯 / 超限 → 当叶子
    f, thr, _ = best_split(X, y, w)
    m = X[:, f] <= thr
    node.feature, node.threshold = f, thr
    node.left  = grow(X[m],      y[m],      ..., depth+1)   # ← 递归
    node.right = grow(X[~m],     y[~m],     ..., depth+1)
    return node

def predict_row(x, node):
    while node.left is not None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.value
```

递归不过是「在子集上重复同一个问题」：**选最优分裂 → 劈成两半 → 对每半再递归**。
`max_depth`/`min_samples_split` 就两个保护递归深度的旋钮（既防过拟合也防栈爆）。

## 6. sklearn 风格封装

框架轨道新增 `src/mini_sklearn/tree/`：

```python
from src.mini_sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

clf = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=0)
clf.fit(X, y)            # tree_（根节点）、classes_、n_features_in_
clf.predict(X)           # 沿根→叶，返回 classes_[多数类]
clf.feature_importances_ # 每特征加权不纯度下降，求和≈1

reg = DecisionTreeRegressor(criterion="squared_error", max_depth=2)
reg.fit(X, y)            # 叶子=加权均值；方差当不纯度
```

- 两个类都缘着 Estimator 协议：`fit`→`self`、`_` 结尾的学习状态、`get_params` 只暴露超参
  （`criterion/max_depth/min_samples_split/random_state`）、predict 前必须 `fit`
  （否则 `check_is_fitted` 抛 `ValueError`）、`n_features_in_`。
- `DecisionTreeClassifier.fit` 额外接受 `sample_weight=None`：当传入时，gini 计数与
  叶子多数类都按**加权**频数来算——这正是下一章 Task 4 的 AdaBoost 重训该树所需的钩子。
- 分裂完全确定性：平局取编号最小的特征/最小的阈值，`random_state` 仅为 API 对齐保留。

## 相对上一章的变化

| | s11（Naive Bayes） | s12（Decision Tree） |
|---|-----|-----|
| 思想 | 先验 × 似然（概率生成） | 递归 if-else 规则（分区） |
| fit 做了什么 | 学每类参数（θ、σ、π） | 长出一棵二叉分裂树 |
| 判断依据 | 后验 `P(y|x) ∝ P(y)P(x|y)` | 特征阈值逐层问路到叶子 |
| 可解释性 | 中（概率打分） | 高（路径就是规则） |
| 非线性边界 | 靠特征的软密度 | 靠递归分区，天然复杂 |
| 推理复杂度 | 一条公式扫一遍，很轻 | O(深度)，深度可被旋钮限制 |

s11 把判断换成「概率」，却仍然要在全部特征上一遍扫过打分；s12 更有"拥"的风格——训练
阶段就把数据**一刀一刀切成小盒子**，predict 只是沿着一条路径走下个深度。决策树也因此
进了"经典非线性模型"梯队（s10–s13）。

## 运行实验

```bash
# 教学轨道：零 import src，手搓递归二分 + gini/熵 + 回归树
python s12_decision_tree/code.py

# 框架轨道：pytest 验证数论正确性 + API 协议（含 sample_weight 钩子）
python -m pytest tests/numerical/test_tree.py tests/api/test_api_tree.py -q
```

输出示例（6 个小二维可分离点几乎是完美分了）：

```text
=== 递归二分决策树 (gini, v0.3.4) ===
数据形状         = (6, 2)
生长出的树：
  [x0 <= -0.79 imp=0.500 n=6]
    [叶子: 值=0 n=2]
    [x0 <= 0.90 imp=0.375 n=4]
      [叶子: 值=1 n=2]
训练集 accuracy = 1.0
特征重要性      = [1. 0.]   求和=1.0000

[PASS] s12: 递归二分（gini/entropy）长成可解释树——一刀弱、递归强
```

## 与 sklearn 对比

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
```

- sklearn 的整套类名与超参数（`criterion`/`max_depth`/`min_samples_split`/`random_state`）
  本节逐一对齐；`feature_importances_` 语义相同（归一化加权不纯度下降）。
- sklearn 用**预排序矩阵 + 数组式扁平树**（`children_left/right`、`feature`、
  `threshold` 的 parallel arrays）并缓存分裂增益，因此能处理几十万样本；本节为教学用
  一把可读的 `TreeNode` 对象递归指针，逻辑更直白、更慢但更好看。
- sklearn 的 `splitter="best"` 与本节完全一样：穷举每个特征的所有中点。sklearn 的 CART
  （回归同样是 MSE）；`criterion="gini"` 是默认，`entropy` 与此一致。

## 失败案例和边界条件

- **未 fit 就 predict**：`check_is_fitted` 拦截，抛 `ValueError`（API 测试锁定）。
- **特征列不匹配**：predict 的 `X` 列数与 `n_features_in_` 不齐 → `ValueError`。
- **纯之退化**：某特征在节点内所有值相同 → 无有效中点；跳过（`xs[i] <= xs[i-1]` 守卫）。
- **退化切分**：threshold 把所有样本归一边 → 空/全；`left_mask.all()/none` 时放弃。
- **过拟合**：`max_depth=None` 会一路砍到纯，把训练噪声也刻进边界——训练误差 0 但泛化差
  。用 `max_depth`/`min_samples_split` 正则化（这正是 s14 交叉验证要找的旋钮）。
- **递归深度**：深度被纯/`min_samples_split`/单类自然掐住；若仍担心可显式设 `max_depth`
  （教学友好，而不是动 `sys.setrecursionlimit`）。

## 教学实现与生产实现的差异

- 教学是对象指针树 × 数姿重排；生产是扁平的一维数组树（`tree_.children_left` 等），
  一次构建多次查询、内存局部性更好。
- 教学每次 `best_split` 都重排每个特征的行；生产预处理一次并把值排序结果缓存复用。
- 生产实现 `predict` 一次性把整批样本广播平移过树，教学逐行走（小数据更清晰）。
- 生产提供 `ccp_alpha`（代价复杂度剪枝）、`splitter="random"`、类别特征等；本节聚焦
  核心的递归划分与两个旋钮。

## 练习题

1. 用 3 行手算 `gini`：对 `counts=[3,1]` 输出 `1-(3/4)²-(1/4)²=0.375`；换成 `entropy`
   再算，对比谁大、为什么。
2. 给 `best_split` 打断点，验证它推荐的中点是「相邻不同值的平均」，且该处
   `split_impurity` 最小（手算一个 3 类 1 维小例）。
3. 把 `max_depth` 从 `None` 一路收到 `1`，在 `s12/code.py` 的可分离数据上观察
   `acc` 与树的大小与训练精度——体会什么是"一刀很弱"、递归才强。
4. 给 `DecisionTreeClassifier.fit` 传一个不均匀 `sample_weight`（比如加倍某一类），
   看叶子值/ gini 是否随之"加权"翻转——它为何是 AdaBoost 重训练树所需的关键钩子？
5. 思考「弯曲边界」：一颗树怎么表达 XOR（特征 0 高+特征 1 高才是一类）这么个非线性
   边界？在纸上画两刀组成一个 2×2 的格子，递归就能做到。

## 下一章

决策树给了我们一棵**可解释**的规则树，但单树深了就过拟合、浅了又欠拟合。下一章
**s13 集成学习**将揭示"三个臭皮匠"的智慧——**用「并行装袋（Bagging）」或「顺序加权
（Boosting）」把一堆弱决策树**拧成一支强分类器**，迈过"单模型"到"集成"的时代。
