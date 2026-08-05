# Learn ML — 从零实现 mini-scikit-learn（NumPy 递进式教程）

从零用 NumPy 搭建一个迷你 scikit-learn，覆盖 `learn-ml/versions.md` 的完整迭代路线：**最小监督学习闭环（v0.1，s01–s04）→ 预处理与线性学习（v0.2，s05–s09）→ 经典非线性模型（v0.3，s10–s13）→ 交叉验证、搜索与 Pipeline（v0.4，s14–s16）→ 无监督学习（v0.5，s17–s18）→ 第三方 Estimator（v0.9，s19）→ 完整 mini-sklearn（v1.0，s20）**。

本教程是 learn-claude-code / learn-llm / learn-seq2seq 系列在"从零拆装 scikit-learn"方向的延伸。核心口号：

```text
One Estimator & NumPy is all you need.
```

不教「调 API」，教「搭 Harness」——从 `fit`/`predict` 最小闭环，逐章演进到 Estimator 协议、数据验证、Transformer、Pipeline 与 Grid Search。**双轨结构**：教学轨道（每章自包含 `code.py`，讲原理）+ 框架轨道（`src/mini_sklearn/` 正式包 + `tests/`）。

| 章节 | 阶段 | 里程碑 | 覆盖版本 | 描述 |
|------|------|--------|---------|------|
| **s00_overview** | — | 全景 | — | 学习路线 + 版本树 + 双轨导览 |
| **s01_minimal_estimator** | 一 | 最小闭环 | v0.1 | 训练状态、`fit`→`self`、`_`-结尾属性 |
| **s02_estimator_protocol** | 一 | 统一接口 | v0.1 | `BaseEstimator`、`get_params/set_params`、`clone` |
| **s03_data_validation** | 一 | 数据契约 | v0.1 | `check_array/check_X_y/check_is_fitted` |
| **s04_evaluation_loop** | 一 | 评估闭环 | v0.1 | `train_test_split` + metrics（MSE/MAE/R²/accuracy） |
| s05…s20 | 二–六 | （后续版次） | v0.2–v1.0 | 本迭代尚未包含；见 `versions.md` |

## Quickstart

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 从 s00 开始阅读
# 每章目录下均有 README.md（中文教程）+ code.py（可运行代码）

# 3. 运行某个章节
python s01_minimal_estimator/code.py

# 4. 运行框架轨道测试
pytest tests/ -q
```

## 链接

- [s00_overview/](s00_overview/)
- [s01_minimal_estimator/](s01_minimal_estimator/)
- [s02_estimator_protocol/](s02_estimator_protocol/)
- [s03_data_validation/](s03_data_validation/)
- [s04_evaluation_loop/](s04_evaluation_loop/)
</content>