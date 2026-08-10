[返回根目录](../README.md)

# s10：Pairwise and KNN——模型也可以不学习显式参数


> 本章口号：**有些模型训练很轻，推理很重。**
>
> 对应 sklearn 模块: `sklearn.metrics.pairwise, sklearn.neighbors`

## 1. 上一章留下的问题

s06–s09 全是**急切学习（eager learning）**：训练期花在拟合**显式参数**上——
s06 的 `LinearRegression` 用伪逆解出 `coef_`，s08 的 `LogisticRegression` 迭代磨出
`coef_ + intercept_`，s09 的 OvR 克隆 k 个这样的分类器。它们的共性：**学完之后，
模型是一串显式的数字**，predict 只是把这串数字代进一个公式扫一遍。

可现实中还有一类完全相反的算法，它们**根本不解出任何参数**：新样本来了没有 公式 可代，
只能回到「老邻居里问一嘴」。s10 就走进这种**懒学习（lazy learning）**的世界——这是
从「学参数」迈向「用数据本身做判断」的第一种非线性模型。

## 2. 本章目标

读完本章，你应该能回答：

- 为什么说 KNN「不学习」？`fit` 到底存了什么？
- 「懒学习」与「急切学习」在训练 / 推理时间上差在哪？
- pairwise 的 `(n_x, n_y)` 距离矩阵怎么用**一条 NumPy 广播**算完，而不是双重 for 循环？
- KNN 的 `fit` 为什么可以只存数据、把 `predict` 一步变成「找邻居 + 聚合」？
- 分类的多数表决与回归的均值，怎么共享同一个选邻居的框架？
- `k`、`metric`、`weights` 三个超参分别控制什么？

核心思想：**fit 只是存数据，predict 才真正搜索——训练很轻、推理很重。**

## 数学直觉

**急切学习**与**懒学习**

```text
急切学习（s06–s09）:  X, y ——(训练，花大算力)-->  coef_  ——(predict 套公式)-->  yhat
懒学习   （s10）   :  X, y ——(fit 原样存档)----->  (X_train, y_train) ——(predict 搜索 k 近邻)-->  yhat
```

对每个待测点 `x_query`，KNN 做两件事：
1) **找邻居**：算 `x_query` 到每个训练点的距离，挑出最小的 `k` 个；
2) **聚合**：分类取 k 个近邻标签的**多数票**，回归取它们的**均值**。

「距离」由选定的 `metric` 决定：欧氏（L2）、曼哈顿（L1）、余弦相似度。`weights` 控制
邻居投票权重：`uniform` 一票一票，`distance` 让越近的邻居越压。

## 数学定义

```text
pairwise 距离矩阵    D[i,j] = || x_i - y_j ||_p               (n_x × n_y)
欧氏(L2)             D[i,j] = sqrt( Σ_d (x_i,d - y_j,d)^2 )
曼哈顿(L1)           D[i,j] = Σ_d |x_i,d - y_j,d|
余弦相似度           S[i,j] = (x_i · y_j) / (||x_i|| · ||y_j||)

KNN 分类  yhat_q  = argmax_c  Σ_{j ∈ N_k(q)} 1[y_j == c]        # k 近邻多数票
KNN 回归  yhat_q  = (1/k) Σ_{j ∈ N_k(q)} y_j                     # k 近邻均值
N_k(q)      = 与 q 距离最接近的 k 个训练样本的下标
```

训练零参数：`fit` 只保存 `(X, y)`。所有「学习」发生在 predict——对每个 query，先算它
到 `n_train` 个点的距离，再取前 k 个。

## 最小实现

```python
class KNNClassifier:
    def fit(self, X, y):               # 懒学习：一行都不优化，只存档
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        out = []
        for x in X:                     # 对每个待测点（predict 才忙）
            d = ((self.X_train - x) ** 2).sum(axis=1)   # 到所有训练点距离
            k_ix = np.argsort(d)[:self.k]               # 最近的 k 个邻居下标
            labels = self.y_train[k_ix]                 # 邻居的标签
            vals, counts = np.unique(labels, return_counts=True)
            out.append(vals[np.argmax(counts)])         # 多数类
        return np.array(out)
```

`predict_point` 里的 `(n_train, d)` 就是一个 mini pairwise：用广播一次算出 query 到
所有训练点的平方距离，`argsort` 切前 k 个。这就是懒学习——**决策规则根本没被「训练」出来，
而是在推理时逐点重建**。

## NumPy 向量化实现

**Python 双循环 → NumPy 广播**，先给框架建 `metrics` 的 pairwise 基元：

```python
# 朴素：双 for，每对样本一次 sqrt(sum(diff^2))
def pairwise_naive(X, Y):
    D = np.empty((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            D[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return D

# 广播：一次 (n_x, n_y, d) 差张量，再沿特征轴开方
def euclidean_distances(X, Y=None):
    X, Y = _prepare(X, Y)               # Y=None -> Y=X
    diff = X[:, None, :] - Y[None, :, :]   # (n_x, 1, d)-(1,n_y,d) → (n_x,n_y,d)
    return np.sqrt((diff ** 2).sum(axis=-1))
```

```python
manhattan：np.abs(diff).sum(axis=-1)
cosine    : 行归一化后再矩阵点积 (X_norm @ Y_norm.T)
```

`X[:, None, :]` 插出一个新轴，和 `Y[None, :, :]` 一起广播成 `(n_x, n_y, d)`——把
「每对样本的距离」完整编码进一条表达式，对 N 个点一次算完。

## scikit-learn 风格封装

框架轨道新增 `src/mini_sklearn/metrics/_pairwise.py` 与 `src/mini_sklearn/neighbors/`：

```python
from src.mini_sklearn.metrics import euclidean_distances, manhattan_distances
from src.mini_sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

D = euclidean_distances(X, X)          # (n_x, n_x)，对角 = 0
clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
clf.fit(X, y)                          # 懒：仅存档 _fit_X / _fit_y
print(clf.predict(X))                  # 搜索最近邻 + 多数票
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X, y_reg).predict(X)           # 邻居均值
```

- `KNeighborsClassifier(n_neighbors, metric, weights)` / `KNeighborsRegressor` 同构。
- `fit` 只存 `_fit_X`/`_fit_y`/`classes_`/`n_features_in_`；`predict` 才找邻居。
- `weights="uniform"`（默认）或 `"distance"`（按 1/d 加权，越近票越重）。
- `check_is_fitted` 保证 predict 前必须 fit，否则 ValueError；特征列不匹配也报错。

## 相对上一章的变化

| | s06–s09（急切学习） | s10（懒学习 KNN） |
|---|-----|-----|
| fit 做了什么 | 解出显式参数（coef_/intercept_/…) | 只存档 `_fit_X`/`_fit_y`，零学习 |
| 训练/推理耗时 | 训练重、推理一行公式 | 训练轻（≈free）、推理最重（逐点搜索） |
| pytest 新增基元 | 线性 / 概率 / 组合 | 成对距离 + k 近邻搜索 + 聚合 |
| 新的 `metrics` | 回归/分类评分 | `_pairwise`：欧氏/曼哈顿/余弦 |
| `predict` | coef_ 点乘算输出 | 按距离 argsort 取 k 近邻再聚合 |

s09 组装出元估计器让基础分类器「被扩展」。s10 反过来：根本不训练任何显式参数，风险更
少地让「模型 = 训练数据本身」。两者都不是「学一个固定的权重」，但懒学习把这做到极致。

## 运行实验

```bash
# 教学轨道：零 import src，手搓 pairwise + 懒 KNN（含回归器）
python s10_neighbors/code.py

# 框架轨道：pytest 验证 pairwise 数值 + neighbors 拟合测试
python -m pytest tests/numerical/test_pairwise.py tests/numerical/test_neighbors.py tests/api/test_api_neighbors.py -q
```

教学示例输出（两个相距很远的二维簇）：

```text
=== pairwise：Python 双循环 vs NumPy 广播 ===
两实现一致? True
euclidean_distances([[0,0],[3,4]]) =
 [[0. 5.]
 [5. 0.]]

=== 懒学习 KNN 分类器 ===
训练数据形状          = (100, 2) （fit 阶段只是原样保存）
可分离二维数据 accuracy = 1.0

[PASS] s10: 懒学习成立——fit 只存数据、predict 才找邻居；KNN 还原二维可分离聚类
```

## 与 sklearn 对比

```python
# sklearn
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
```

- sklearn 的 pairwise 同样支持 `Y=None` 自我配对，对角为 0；本章的签名一致。
- sklearn 的 `KNeighborsClassifier` 也用 `n_neighbors`/`metric`/`weights`，语义对等——不过
  生产端默认 `metric='minkowski'`、会训 KD-Tree / Ball-Tree 来加速搜索，而这里直接扫描
  全训练集（对教学数据足够）。
- sklearn 提供 `kneighbors()`（返回距离 + 邻居下标）、`kneighbors_graph`，本章只取最朴素的
  search + majority/mean。
- sklearn 默认用 `metric='minkowski'` 且 `p=2`（当于欧氏）、`cosine` 也做行归一化——本章的
  `euclidean` / `manhattan` 与它语义对齐，只是没实现 p 阶的 Minkowski。

## 失败案例和边界条件

- **特征列不匹配**：predict 的 `X` 列数与 `n_features_in_` 不齐 → `ValueError`（沿用框架校验）。
- **未 fit 就 predict**：靠 `_fit_X` 未设，`check_is_fitted` 直接报 `ValueError`，api 测试锁定。
- **邻居被 zero 距离**：老数据自己 predict 自己时，k 最近常值 0；距离加权按 `1/d` 会发 `inf`，
  故对 d=0 夹到极小底限 1e-12，避免权重爆炸。
- **ties（平票）**：多数票用 `np.argmax` 取最先出现的类，确定性、不随机。
- **大 O**：懒模型 intrinsics 是推理，扫描法是 O(n_query × n_train)；生产用树结构（KD/Ball-
  Tree）把均摊降到 sub-linear，这是最待改进的地方。
- **距离尺度**：特征单位不一致时，欧氏距离会被量纲大的轴支配；生产先标准化，教学暂时
  用同尺度合成数据绕过。

## 教学实现与生产实现的差距

- 教学 `euclidean` 用全扫描 `(n_q × n_train)` 的距离矩阵；生产用树约化。懒学习的大样本
  性能瓶颈就在这。
- 教学 `weights` 只实现 `uniform` 与 `distance`（1/d）；生产还支持 `callable` 自定义权重
  与 `p` 阶 Minkowski。
- 教学只按欧氏/曼哈顿/余弦三种指标；生产支持数十种以及距离化内核化。
- 教学 `fit` 只存 ndarray；生产还缓存树结构、记录 `classes_` 与哑编码等。

## 练习题

1. 把 pairwise_naive（双循环）改写作单个 NumPy 广播，跟踪形状从 (n_x,1,d) 到 (n_y,1,d) 展开的过程。
2. 试 `k=1`：KNN 分类等价于"最近邻"，等价于 `argmin(distances)`。用 `np.argmin` 复写 `predict` 一个简版,验证等价。
3. 给 KNN `predict_point` 加上 `weights`：用 `uniform` 和 `1/d`（注意 d=0 兜底）两种加权分别跑回归，看在强噪数据下谁会更好。
4. 把 `n_neighbors` 从 1 改成整个训练集大小，KNN 分类会退回成什么？（全部多数类）
5. sklearn 的 `KNeighborsClassifier` 默认 `metric='minkowski', p=2`；试 `manhattan` 距离读一个 blob，MNIST 手写数字下哪个效果更好？为什么？

## 下一章

KNN 靠「距离」判断一切，但距离这种实值比较，在类别多、维数高时会变得贵且不稳定。下一章
s11 **Naive Bayes** 把是非判断换成概率推断——从「打听邻居」转向「先验 × 似然」，用到毫不
可言的距离，只需算一组朴素的条件概率。