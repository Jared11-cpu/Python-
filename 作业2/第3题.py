from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 读取数据：CSV 与本程序放在同一目录，Sales 的单位为千件。
data_path = Path(__file__).with_name("Carseats.csv")
data = pd.read_csv(data_path)
y = data["Sales"]
print(f"样本数：{len(data)}")

# 2. 设置自变量。ShelveLoc 有三类，以 Bad 为基准建立两个虚拟变量。
# Bad 对应 (0, 0)，Good 对应 (1, 0)，Medium 对应 (0, 1)。
x = data[["Price", "Income", "Advertising"]].copy()
x["ShelveLoc[Good]"] = (data["ShelveLoc"] == "Good").astype(int)
x["ShelveLoc[Medium]"] = (data["ShelveLoc"] == "Medium").astype(int)
x = sm.add_constant(x)  # 添加截距项

# 3. 使用普通最小二乘法拟合，输出系数、标准误、t 检验等结果。
model = sm.OLS(y, x).fit()
print("\n（1）模型拟合报告")
print(model.summary())
print(f"R² = {model.rsquared:.4f}，调整后的 R² = {model.rsquared_adj:.4f}")
print("ShelveLoc 的基准组：Bad")

# 4. Good 系数表示其他变量相同时，Good 与 Bad 的预测销量差。
good = model.params["ShelveLoc[Good]"]
print("\n（2）ShelveLoc[Good] 的含义")
print(f"价格、收入和广告预算相同时，Good 组的预测销量比 Bad 组高 {good:.3f} 千件。")
print(f"约合 {good * 1000:.0f} 件；这是条件相关关系，不能直接认定为因果关系。")

# 5. VIF = 1 / (1 - R_j²)。辅助回归包含截距，但不报告截距的 VIF。
vif = pd.DataFrame({
    "变量": x.columns[1:],
    "VIF": [variance_inflation_factor(x.values, i) for i in range(1, x.shape[1])],
})
print("\n（3）VIF 与多重共线性")
print(vif.round(4).to_string(index=False))
print("ShelveLoc 的两个虚拟变量分别计算 VIF，5 作为经验参考值。")
if vif["VIF"].max() < 5:
    print("各项 VIF 均小于 5，无明显多重共线性风险。")
else:
    print("存在 VIF 不小于 5 的变量，需关注多重共线性。")
