# 作业五 · 无监督学习

## ① WCSS 的成对距离等价形式

设簇内有 $n$ 个样本，质心为 $\mu=\frac1n\sum_i x_i$。展开到质心的平方距离，利用 $\sum_i x_i=n\mu$：

$$
\sum_i\lVert x_i-\mu\rVert^2
=\sum_i\lVert x_i\rVert^2-n\lVert\mu\rVert^2.
$$

对成对距离求和时，每个样本的平方范数分别在两个指标上出现 $n$ 次：

$$
\sum_i\sum_j\lVert x_i-x_j\rVert^2
=2n\sum_i\lVert x_i\rVert^2-2\left\lVert\sum_i x_i\right\rVert^2
=2n\left(\sum_i\lVert x_i\rVert^2-n\lVert\mu\rVert^2\right).
$$

双重求和除以 $2n$ 就得到 WCSS。每一对不同样本按两个顺序都算了一次。

## ② Ward 聚类合并的方差增加量

令 $N=n_A+n_B$，合并后的质心是 $\mu=(n_A\mu_A+n_B\mu_B)/N$。对 $A$ 中的点，把 $x-\mu$ 分成 $(x-\mu_A)+(\mu_A-\mu)$。平方求和时，交叉项因 $\sum_{x\in A}(x-\mu_A)=0$ 消失。对 $B$ 同理。

$$
\Delta ESS=ESS(A\cup B)-ESS(A)-ESS(B)
=n_A\lVert\mu_A-\mu\rVert^2+n_B\lVert\mu_B-\mu\rVert^2.
$$

代入 $\mu_A-\mu=(n_B/N)(\mu_A-\mu_B)$ 和 $\mu_B-\mu=-(n_A/N)(\mu_A-\mu_B)$，合并两项系数可得：

$$
\Delta ESS=\frac{n_A n_B}{n_A+n_B}\lVert\mu_A-\mu_B\rVert^2.
$$

## ③ 维度灾难与前置处理

如果许多坐标大致独立、尺度相近，两点的平方距离就是许多项相加。维度增加时，和的均值通常按维度增长，标准差只按维度的平方根增长，远近距离的相对差异随之缩小。K-Means 的平方距离容易被无关特征主导。对 DBSCAN 来说，固定样本量下局部邻域变稀疏，ε 稍小就会留下许多噪声点，调大又容易把簇连在一起。

先标准化以减轻量纲影响，再结合特征筛选、PCA 或 TruncatedSVD 压缩无关维度。方向更重要时可尝试余弦距离，稀疏二值特征可考虑 Jaccard。降维后仍需检查邻域，因为最大方差方向可能与分群无关。

## ④ Iris 聚类实战

对 150 条 Iris 记录的四个测量特征做 Z-score。K-Means 比较 $K=2$ 到 $8$，每个 $K$ 用 20 次初始化。品种标签只用于事后计算 ARI。

![手肘曲线与按簇排序的轮廓刀片图](kmeans_diagnostics.png)

将 WCSS 曲线归一化后，离首尾连线向下最远的点是 **K=3**。平均轮廓系数最高的是 **K=2（0.5817）**，所以本次按平均轮廓系数选两簇。图中每条刀片的高度对应簇大小，虚线为平均轮廓系数。

![原特征空间拟合与 PCA 二维拟合的聚类对比](pca_comparison.png)

图的左列先在完整特征空间聚类，再投影到 PCA 平面；右列直接在二维主成分上聚类。Iris 前两个主成分解释 95.81% 的总方差，K-Means 在选定的 $K=2$ 下，两种拟合的 ARI 均为 0.5681。

DBSCAN 固定 `min_samples=5`，各空间分别取第 5 近邻距离（含自身）的第 90 百分位为 ε。四维和 PCA 二维都得到两簇，ARI 从 0.5503 降到 0.5379，二维图便于查看边界。加入 100 列独立正态噪声后，相对距离跨度从 1.3915 缩到 0.1694。PCA 拉开了距离，但前两个主成分只解释 6.79% 的总方差，两种加噪空间在该 ε 规则下都只得到一簇，ARI 接近 0。

相对距离跨度为成对距离的第 90 与第 10 百分位之差除以中位数。完整数值见 [实验结果表](results.md)。

## 复现

```sh
python -m pip install -r requirements.txt
python -W error analysis.py
```

脚本读取 [scikit-learn 内置 Iris 数据](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)，生成两张 PNG 图和 `results.md`。随机种子为 42。
