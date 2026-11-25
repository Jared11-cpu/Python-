# 作业五 · 无监督学习

[下载 index.html](index.html) 后用浏览器打开。页面按课件题号收录 WCSS 证明、Ward 合并代价推导、维度灾难分析和 Iris 聚类实验。

实验可以切换 K，查看 WCSS、各簇的轮廓刀片图，以及四维拟合和 PCA 二维拟合的散点图。DBSCAN 部分可以调节近邻距离百分位，并对照原始 Iris 与加入 100 列噪声后的结果。

## 复现

```sh
python -m pip install -r requirements.txt
python -W error analysis.py
```

脚本读取 scikit-learn 内置的 Iris 数据，将实验结果写进 `index.html`，同时在终端打印指标。本次运行使用 Python 3.13、scikit-learn 1.9.1，随机种子为 42。公开的品种标签只用于事后计算 ARI。
