from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读数据
data_path = Path(__file__).with_name("Carseats.csv")
data = pd.read_csv(data_path)
y = data["Sales"]
print(f"样本数：{len(data)}")

# 自变量，ShelveLoc 做成虚拟变量（以 Bad 为基准）
x = data[["Price", "Income", "Advertising"]].copy()
x["ShelveLoc[Good]"] = (data["ShelveLoc"] == "Good").astype(int)
x["ShelveLoc[Medium]"] = (data["ShelveLoc"] == "Medium").astype(int)
x = sm.add_constant(x)

# 拟合
model = sm.OLS(y, x).fit()
print("\n（1）模型拟合报告")
print(model.summary())
print(f"R² = {model.rsquared:.4f}，调整后的 R² = {model.rsquared_adj:.4f}")
print("ShelveLoc 的基准组：Bad")

# Good 的系数含义
good = model.params["ShelveLoc[Good]"]
print("\n（2）ShelveLoc[Good] 的含义")
print(f"其他条件相同的情况下，Good 比 Bad 的预测销量高 {good:.3f} 千件（大约 {good*1000:.0f} 件）。")
print("这是相关关系，不能直接说成因果。")

# 算 VIF
vif = pd.DataFrame({
    "变量": x.columns[1:],
    "VIF": [variance_inflation_factor(x.values, i) for i in range(1, x.shape[1])],
})
print("\n（3）VIF")
print(vif.round(4).to_string(index=False))
print("一般看 5 作为参考，这里都比较低。")
if vif["VIF"].max() < 5:
    print("没有明显的多重共线性问题。")
else:
    print("有变量的 VIF 偏高，需要注意。")
