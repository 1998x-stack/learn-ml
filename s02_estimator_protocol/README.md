# s02：Estimator Protocol——新增模型，不改变调用方式

[返回根目录](../README.md)

> 本章口号：新增一个 Estimator，只改模型内部，不改调用框架。
>
> 对应 sklearn 模块: `sklearn.base`

## 1. 问题

上章（s01）我们把 `fit(X, y) → self` 与 `predict(X) → array` 的「最小闭环」立了起来，并用 `_` 后缀标出了训练产物。但它留下一个尴尬的事实：

每个模型都可以定义自己的 `__init__` 参数，而**框架层**（GridSearch、Pipeline、模型持久化）根本不知道你的模型当初是用什么参数构造的。

假设我们要跑超参数搜索：搜索器想知道一个模型可调的参数有哪些、当前值是多少、以及怎样在给定一组新参数时重建一个一模一样的模型。若没有统一约定，搜索器就无从下手。

本章把这个「统一约定」固化成 **Estimator 协议**：`get_params` / `set_params` / `clone`，配上 `BaseEstimator` 与若干 Mixin。

## 2. 本章目标

读完本章，你应该能回答：

- 超参数（hyperparameter）在 `get_params` 里是如何被发现、读出、改写的？
- `clone` 为什么是「重建」，而不是「复制」？
- `BaseEstimator.get_params` 如何只凭 `__init__` 签名就工作，而不用每个模型手写？
- Mixin（`RegressorMixin` / `ClassifierMixin` / `TransformerMixin`）各自补全了什么？
- 「新增模型，不改变调用框架」为什么是 sklearn 生态能自由扩展 API 的地基？

我们开始在**框架轨道**上落地第一个正式包 `src/mini_sklearn/`，并用 pytest 验证其行为。

## 3. 数学直觉

机器学习里，参数分两类：

- **学习参数 / 训练状态**：`fit` 从数据里学出来的，带 `_` 后缀，如 `constant_`、`coef_`。
- **超参数**：调用者构造模型时给定的、**不随训练变化**的旋钮，如 `strategy`、`offset`。

Estimator 协议做的，就是把「超参数」变成一个模型对框架层可见、可读、可改、可重建的公共契约：

```text
构造 → get_params 读出超参数
     → set_params 修改超参数
     → clone 用当前超参数重建一个"全新未训练"的模型
```

框架层不需要知道任何具体模型的细节，只需调用这四个字面量约定的接口，就能一致地管理任意模型。

## 4. 数学定义

**Estimator 协议**（与 sklearn 的 `sklearn.base` 对齐）：

```text
get_params(deep=True) -> dict
    返回超参数词典：{ 参数名: 当前值 }

set_params(**params) -> self
    就地更新超参数，未知参数名抛 ValueError；返回 self（链式）

clone(estimator) -> estimator
    用 estimator.get_params() 重建 estimator.__class__ 的一个全新实例
```

其中 `get_params` 的「超参数」定义是**不变量**：

```text
训练状态：fit 后出现，带尾部下划线，见 s01
超参数：构造期写入，不带下划线，get_params 只返回这些
```

## 5. 最小实现

核心思想是**内省**：`get_params` 不写在每个模型里，而是靠 `inspect.signature(self.__init__)` 自动发现超参数名，再反射取值。

```python
class BaseEstimator:
    def get_params(self, deep=True) -> dict:
        names = [k for k in signature(self.__init__).parameters if k != "self"]
        return {name: getattr(self, name) for name in names}

    def set_params(self, **params) -> "BaseEstimator":
        for key, value in params.items():
            if not hasattr(self, key):
                raise AttributeError(f"Invalid parameter {key!r}")
            setattr(self, key, value)
        return self
```

`clone` 是工厂：把 `get_params()` 的结果原样喂回类构造函数，得到**全新**对象。

```python
def clone(estimator):
    return estimator.__class__(**estimator.get_params())
```

## 6. NumPy 向量化实现

本章的核心不是数值运算，而是「反射 + 字典」的元编程。但为了演示协议能装上真实模型，我们仍以 `MeanRegressor` 为例：

```python
class RegressorMixin:
    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        u = float(np.sum((y - y_pred) ** 2))          # 残差平方和
        v = float(np.sum((y - np.mean(y)) ** 2))      # 总平方和
        return 1.0 - u / v                            # R²

class MeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.constant_ = float(np.mean(y)) + self.offset
        self.n_features_in_ = X.shape[1]
        return self
```

预测依旧 `np.full(...)`，而 `score` 用整条数组的方和一次性算 R²——同样保持「向量化而非 for 循环」的惯例。

## 7. scikit-learn 风格封装

框架轨道的正式封装放在 `src/mini_sklearn/base.py`：`BaseEstimator`、`ClassifierMixin`、`RegressorMixin`、`TransformerMixin`、`clone`。调用者只需：

```python
from src.mini_sklearn.base import BaseEstimator, RegressorMixin

class MyModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.5, max_iter=100):
        self.alpha = alpha          # 超参数
        self.max_iter = max_iter

model = MyModel(alpha=0.1)
model.get_params()   # {'alpha': 0.1, 'max_iter': 100}
model2 = clone(model)
```

`get_params` 两侧（框架轨道 vs 教学轨道）行为一致，只是教学轨道把它内联进 `code.py` 讲原理。

## 8. 相对上一章的变化

上一章（s01）首次引入训练状态与 `_` 后缀，但它只有一个约定 `fit→self`，**没有统一的参数反射接口**，模型无法被框架层管理。

本章在 s01 基础上，把「协议」从「有没有 `fit`」升级为「有没有标准参数接口」：

- 教学轨道 `code.py` 新增 `BaseEstimator`、`RegressorMixin`、`clone`，并用 `MeanRegressor` 继承协议。
- 框架轨道新建正式包 `src/mini_sklearn/`（`__init__.py` 置 `__version__`），`base.py` 承载正式实现。
- 框架轨道还引入 **pytest 测试** `tests/api/test_api_base.py`（此时 s01 没有任何测试）——协议开始被机器验证而非仅靠人读。

## 9. 运行实验

```bash
# 教学轨道：自包含可运行
python s02_estimator_protocol/code.py

# 框架轨道：pytest 验证 BaseEstimator / clone
python -m pytest tests/api/test_api_base.py -q
```

示例输出（`code.py` 种子 `np.random.default_rng(0)`）：

```text
get_params: {'strategy': 'mean', 'offset': 0.1}
predict: [2.2226823 2.2226823 2.2226823]
克隆是全新对象: True | 超参数一致: True
score (R²): -0.021
[PASS] s02: get_params / set_params / clone / score 一致，Estimator 协议成立
```

测试预期输出：

```text
3 passed in 0.03s
```

## 10. 与 sklearn 对比

本章的 `BaseEstimator` / `clone` 是 `sklearn.base` 的缩小版：

```python
# sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
model = MeanRegressor()            # 继承 BaseEstimator, RegressorMixin
copy = clone(model)                # 深度复制超参数，重建未训练实例
# 本章
from src.mini_sklearn.base import BaseEstimator, RegressorMixin, clone
copy = clone(model)
```

与 sklearn 的差异（我们刻意简化）：

- sklearn 的 `get_params` 对「嵌套估计器」支持 `deep=True`（参数名带 `__`），我们只做扁平超参数，`deep` 参数兼容签名但不展开嵌套。
- sklearn 的 `set_params` 在完全未知参数时抛 `ValueError`，并容忍「另一套补丁处理」；我们直接 `hasattr` 检查。
- sklearn 的 `clone` 会做 `safety of copy`（缺省拷贝），我们直接反射重建。

`get_params` / `set_params` / `clone` 的类型签名（一式两轨）相同：

```text
get_params(deep=True) -> dict       # 超参数名 → 值
set_params(**params) -> self (Estimator)
clone(estimator) -> estimator（全新）
```

## 11. 失败案例和边界条件

- **未知参数 `set_params`**：对不存在的超参数调用就抛 `AttributeError`/`ValueError`，避免静默写错属性名。
- **瞎拼的 `__init__`**：若模型在 `__init__` 里把训练产物也塞进去（如 `self.coef_`），它会混入 `get_params` 返回超参数，破坏协议。生产实现用 `_`-后缀约定就隐式隔离——这也是 `fit` 才写学习结果的原因。
- **可变默认值**：若超参数是 list/dict 等可变对象且被共享，clone 不会深度复制——sklearn 有专门 `safe_clone` 的深度复制，这里简化。
- **构造器必须接受引用的超参数**：`clone` 用 `**get_params()` 打发，若某个超参数名写不进 `__init__`，重建会直接 `TypeError`。

## 12. 教学实现与生产实现的差距

- 教学 `base.py` / `code.py` 内联版只有几十行，而 sklearn 的 `base.py` 包含 `clone` 的深度拷贝、`deep=True` 嵌套参数、`is_clusterer` 等 `_estimator_type` 的大系统。
- sklearn 的 `BaseEstimator` 几乎只负责 `get_params/set_params/repr`；真正的 `fit` 通过 Mixin 约定继承，不强制每个模型都要实现。
- 我们没有处理特征名 `feature_names_in_`、参数 `repr`、cluster 参数读写保护等工程细节。
- `ClassiferMixin` / `RegressorMixin` 在 sklearn 中还参与 `is_classifier/is_regressor` 判定与 `score` 默认实现，本轨只做 `_estimator_type` 极简版。

这些是「教学优先」的取舍——协议的行为是核心，工程健壮性留给正式 sklearn。

## 13. 练习题

1. 给 `MeanRegressor` 保留 `strategy` 但移除 `offset`，重新跑 `code.py`，观察 `get_params` 输出变化，确认超参数集与 `__init__` 一一对应。

2. 往框架轨道新增一个 `_Estimator` 子类字段并在 `tests/api/test_api_base.py` 里断言 `get_params` 只含超参，若含 `_` 会怎样？验证后恢复。

3. 为 `BaseEstimator` 的 `set_params` 传入一个不存在参数（如 `model.set_params(magic=1)`），确认按预期抛错。

4. 用 `sklearn.base.clone` 对比本地 `clone` 对嵌套 Pipeline 的行为差异（`deep=True`），思考为什么 sklearn 需要 `nested_params`。

## 14. 下一章

现在任意模型都有了统一超参数接口，但 `fit`/`predict` 仍在没做任何输入校验——传入字符串、`NaN`、甚至 `y` 为 2D，模型很可能静默返回垃圾。

下一章 `s03_data_validation` 引入 `check_array` / `check_X_y` / `check_is_fitted`，把「数据契约」固化，让每个 Estimator 在 `fit`/`predict` 前都用同一套校验守卫自己的输入。
