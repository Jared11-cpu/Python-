from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv(Path(__file__).with_name("Carseats.csv"))

# Bad 为基准组，另外两类用虚拟变量表示。
x = data[["Price", "Income", "Advertising"]].copy()
x["ShelveLoc[Good]"] = (data["ShelveLoc"] == "Good").astype(int)
x["ShelveLoc[Medium]"] = (data["ShelveLoc"] == "Medium").astype(int)
x = sm.add_constant(x)
model = sm.OLS(data["Sales"], x).fit()
print(model.summary())

good = model.params["ShelveLoc[Good]"]
print("\n基准组：Bad")
print(f"价格、收入和广告预算相同时，Good 组的预测销量比 Bad 组高 {good:.3f} 千件。")

# 计算各解释变量的 VIF，辅助回归保留截距。
vif = pd.Series(
    [variance_inflation_factor(x.values, i) for i in range(1, x.shape[1])],
    index=x.columns[1:],
    name="VIF",
)
print("\n", vif.round(4))
if vif.max() < 5:
    print("各项 VIF 均小于 5，无明显多重共线性风险。")
else:
    print("存在 VIF 不小于 5 的变量，需关注多重共线性。")
