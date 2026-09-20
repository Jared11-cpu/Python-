# 作业2：线性回归

- [第1题：残差性质的证明](作业2/第1题.md)
- [第2题：R² 与调整后的 R²](作业2/第2题.md)
- [第3题：Carseats 回归分析](作业2/第3题.py)
- [第3题运行结果](作业2/第3题_运行结果.txt)

第3题使用 `Price`、`Income`、`Advertising` 和 `ShelveLoc` 解释 `Sales`。程序输出模型拟合报告、货架位置的基准组、`ShelveLoc[Good]` 的含义及各解释变量的 VIF。

## 运行第3题

在仓库根目录执行：

```bash
python -m pip install -r 作业2/requirements.txt
python 作业2/第3题.py
```

已在 Python 3.13 下运行验证。程序读取同目录的 `Carseats.csv`，结果直接显示在终端；仓库中的 `第3题_运行结果.txt` 保存了一次运行结果。

## 数据来源

Carseats 是 ISLR 中的模拟数据，共有 400 家门店的观测。`Sales` 的单位是千件，`Income` 和 `Advertising` 的单位是千美元。数据说明见 [ISLR 文档](https://search.r-project.org/CRAN/refmans/ISLR/html/Carseats.html)，CSV 取自 [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv)。

VIF 使用 statsmodels 计算，函数说明见 [variance_inflation_factor](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)。
