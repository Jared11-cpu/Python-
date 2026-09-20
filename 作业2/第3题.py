from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 读入数据
file_path = Path(__file__).parent / "Carseats.csv"
data = pd.read_csv(file_path)
y = data["Sales"]

# Bad 作为参照组
x = data[["Price", "Income", "Advertising"]].copy()
x["ShelveLoc[Good]"] = (data["ShelveLoc"] == "Good").astype(int)
x["ShelveLoc[Medium]"] = (data["ShelveLoc"] == "Medium").astype(int)
x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
print("样本数：", len(data))
print(model.summary())

good_coef = model.params["ShelveLoc[Good]"]
print("\nGood 的回归系数：", round(good_coef, 4))
print("按模型估计，其他变量不变时，Good 比 Bad 的销量高约",
      round(good_coef, 3), "千件。")

# 计算各自变量的 VIF，不计算常数项
vif_result = pd.DataFrame({
    "变量": x.columns[1:],
    "VIF": [
        variance_inflation_factor(x.values, i)
        for i in range(1, x.shape[1])
    ]
})

print("\nVIF 结果：")
print(vif_result.round(4).to_string(index=False))
