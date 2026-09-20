"""作业2，第3题：Carseats 多元线性回归。

安装依赖：python -m pip install -r requirements.txt
运行：python 第3题.py

数据说明：https://search.r-project.org/CRAN/refmans/ISLR/html/Carseats.html
CSV 来源：https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():
    folder = Path(__file__).resolve().parent
    data_path = folder / "Carseats.csv"
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        url = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"
        data = pd.read_csv(url)
        data.to_csv(data_path, index=False)

    columns = ["Sales", "Price", "Income", "Advertising", "ShelveLoc"]
    data = data[columns].copy()
    if data.isna().any().any():
        raise ValueError("建模变量中有缺失值，请先检查数据。")
    if set(data["ShelveLoc"].unique()) != {"Bad", "Good", "Medium"}:
        raise ValueError("ShelveLoc 应包含 Bad、Good 和 Medium 三类。")

    # 以 Bad 为基准，只保留 Good 和 Medium 两个虚拟变量。
    x = data[["Price", "Income", "Advertising"]].astype(float).copy()
    x["ShelveLoc[Good]"] = (data["ShelveLoc"] == "Good").astype(float)
    x["ShelveLoc[Medium]"] = (data["ShelveLoc"] == "Medium").astype(float)
    x = sm.add_constant(x)
    model = sm.OLS(data["Sales"], x).fit()

    # VIF 的辅助回归保留截距，但不把截距作为解释变量报告。
    vif = pd.DataFrame({
        "变量": x.columns[1:],
        "VIF": [
            variance_inflation_factor(x.to_numpy(), i)
            for i in range(1, x.shape[1])
        ],
    })

    beta = model.params
    good = beta["ShelveLoc[Good]"]
    lower, upper = model.conf_int().loc["ShelveLoc[Good]"]
    rss = float(np.sum(model.resid ** 2))
    rse = float(np.sqrt(model.mse_resid))
    max_vif = float(vif["VIF"].max())
    if max_vif < 5:
        vif_comment = "各项 VIF 都小于 5，没有发现明显的多重共线性风险。"
    else:
        vif_comment = "有变量的 VIF 达到或超过 5，需要进一步检查共线性。"

    equation = (
        f"Sales = {beta['const']:.4f}"
        f" {beta['Price']:+.4f} × Price"
        f" {beta['Income']:+.4f} × Income"
        f" {beta['Advertising']:+.4f} × Advertising"
        f" {good:+.4f} × I(ShelveLoc=Good)"
        f" {beta['ShelveLoc[Medium]']:+.4f} × I(ShelveLoc=Medium)"
    )
    report = [
        "第3题：Carseats 多元线性回归",
        "",
        f"样本数：{int(model.nobs)}；残差自由度：{int(model.df_resid)}。",
        "Sales 表示销量，单位为千件；Income 和 Advertising 的单位为千美元。",
        "",
        "一、模型拟合报告",
        model.summary().as_text(),
        "",
        "拟合方程：",
        equation,
        f"RSS = {rss:.4f}，残差标准误 = {rse:.4f} 千件。",
        f"R² = {model.rsquared:.4f}，调整后的 R² = {model.rsquared_adj:.4f}。",
        f"模型解释了样本中约 {model.rsquared * 100:.2f}% 的销量变异。",
        "",
        "二、ShelveLoc 的基准组和系数含义",
        "基准组是 Bad。该组的两个虚拟变量都取 0，其影响包含在截距中。",
        f"ShelveLoc[Good] 的系数为 {good:.4f}。在价格、收入和广告预算相同的条件下，",
        f"货架位置为 Good 的门店，预测销量比 Bad 的门店平均高 {good:.4f} 千件，",
        f"约合 {good * 1000:.0f} 件。这里的比较对象是 Bad，并不是 Medium。",
        f"该系数的 95% 置信区间为 [{lower:.4f}, {upper:.4f}] 千件，",
        f"p 值为 {model.pvalues['ShelveLoc[Good]']:.3e}。",
        "从经营角度看，较好的陈列位置与较高的销量相关，值得在陈列安排中考虑。",
        "不过，这一回归结果本身不能保证仅调整货架位置就能带来同样的销量增幅。",
        "",
        "三、VIF 与多重共线性",
        "VIF = 1 / (1 - R_j²)，其中 R_j² 来自用其余解释变量预测第 j 个变量的辅助回归。",
        vif.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "ShelveLoc 有三个水平，所以分别列出两个虚拟变量的 VIF。",
        "它们是设计矩阵各列的 VIF，并不是 ShelveLoc 整体的广义 VIF。",
        f"最大 VIF 为 {max_vif:.4f}。{vif_comment}",
        "5 是常用的经验参考值，不是绝对界限。",
    ]
    report_text = "\n".join(line.rstrip() for line in "\n".join(report).splitlines()) + "\n"
    print(report_text)
    (folder / "第3题_运行结果.txt").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
