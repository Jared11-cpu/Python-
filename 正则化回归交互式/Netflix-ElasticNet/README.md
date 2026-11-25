# 正则化回归交互式

用 Elastic Net 分析 Netflix 电影时长，并和 Ridge、Lasso 做比较。运行 `app.py` 可以调节参数，查看误差和系数的变化。

## 这次分析的问题

电影的年份、国家、类型和演员数量，能提供多少关于时长的信息？这里把电影分钟数作为预测目标，先用平均时长做一个简单预测，再看正则化回归能改善多少。

数据来自 Kaggle 的 [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)，下载日期为 2026-11-25。这份数据记录的是影视目录，上映年份截至 2021 年。`rating` 一列是 TV-MA、PG-13 这样的内容分级。

原始文件有 8,807 条记录，其中电影 6,131 条。电影时长以分钟计，电视剧则按季数记录，所以这里只分析电影。去掉 3 条缺失时长的记录后，剩下 6,128 部。

## 运行

建议 Python 3.11 或更新版本。在本目录执行：

```bash
python -m pip install -r requirements.txt
python download_data.py
python analysis.py
python verify.py
python -m streamlit run app.py
```

原始 CSV 不提交到仓库，下载脚本从 Kaggle 公开入口取回。数据校验值和运行版本记录在 `results/manifest.json`，便于检查日后数据源是否变化。交互页面需要先下载数据；查看已有报告和结果表不需要启动页面。

## 分析方法

1. 删除重复的 `show_id`，保留时长能解析为正数分钟的电影。较长的电影保留，不因其离均值远就直接删掉。
2. 用 `release_year`、演员人数、导演人数、国家数、类型数作为数值特征；主要国家、主要类型和内容分级作为类别特征。多个国家或类型以原字段中的第一个为“主要”，这是一个简化，原始顺序未必代表真实主次。
3. 按 80% / 20% 随机划分训练和测试数据，随机种子固定为 42。测试集不参与参数选择。这个划分评估的是同一目录分布内的插值能力，不能直接说明模型对未来年份新片的预测能力。
4. 数值缺失用训练折中位数填补，类别缺失记作 `Unknown`。类别独热编码，并将训练折中少于 20 次的类别合并。所有编码后的特征再做 Z-score 标准化，截距不惩罚。填补、编码和缩放都在 Pipeline 内随每个训练折重新拟合。
5. 三种回归使用相同的打乱后 10 折划分。用训练折内的均方误差选择参数，而不比较测试集来挑选最终模型。使用 GridSearchCV 包住整个 Pipeline，而不是先对全部训练数据标准化再调用内部 CV；这样能避免验证折信息进入预处理。
6. Elastic Net 同时搜索 `alpha` 和 `l1_ratio`。另报告 1-SE 规则版本：CV MSE 不超过最小值加该点标准误的候选中，优先选非零系数最少的模型，再优先更大的 alpha，最后比较 CV MSE。标准误近似为 10 折误差标准差除以 √10，不是独立重复试验下的严格置信区间。

Ridge 的 alpha 在 0.01–100,000 间取 22 个对数等距值；Lasso 和 Elastic Net 在 0.01–100 间取 13 个值。Elastic Net 的 l1_ratio 候选为 0.1、0.5、0.9。初始 Ridge 网格上界为 100，因最优值位于边界，已扩大范围复查。复现本次运行版本可使用 `requirements-lock.txt`；普通安装使用 `requirements.txt`。

scikit-learn 的 Elastic Net 目标函数是：

\[
\frac{1}{2n}\|y-X\beta\|_2^2+
\alpha\rho\|\beta\|_1+
\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2
\]

其中 `rho = l1_ratio`。L1 部分能把部分系数压到零，L2 部分能缓和相关变量下的系数不稳定。`alpha` 越大，整体惩罚越强。Ridge 与 Elastic Net 在 scikit-learn 中的损失归一化方式不同，因此相同 alpha 数值不能理解为相同的惩罚力度。

## 文件说明

| 文件 | 用途 |
|---|---|
| `analysis.py` | 数据清洗、10 折调参、1-SE 选择、测试与绘图 |
| `app.py` | 调节参数、查看验证误差和分析结果 |
| `download_data.py` | 下载 Kaggle 原始 CSV |
| `分析报告.md` | 数值结果、系数解释和局限 |
| `results/metrics.csv` | 预测误差、R²、参数与非零系数数目 |
| `results/cross_validation.csv` | 各参数组合的 CV MSE 与标准误 |
| `results/coefficients.csv` | 正式 Elastic Net 模型的标准化系数 |
| `results/coefficient_paths.svg` | 三种模型的系数路径 |
| `results/diagnostics.svg` | 时长分布及测试集预测散点图 |
| `results/predictions.csv` | 测试集预测及残差，含样本编号 |
| `results/train_ids.csv` | 训练样本编号，可与测试编号核对 |
| `verify.py` | 复核划分、误差计算、1-SE 条件和两个交互滑块 |

交互试验另从正式训练集划出 25% 作为验证集。滑块显示的是内部验证误差，不是正式测试误差，避免反复拖动参数时逐渐把测试集当成训练依据。

## 结果如何理解

RMSE 和 MAE 的单位均为分钟，越小越好。R² 与“预测训练集平均时长”的基线一起看，不能只看某一个分数。非零系数按绝对值大于 `1e-8` 计数，这里统计的是编码后的列，而非原始字段数。

系数表示特征标准化后变化一个标准差时的条件关联。类别变量使用完整独热编码，正则化会影响冗余编码下的系数分配；不要把单个国家系数理解成相对某个未列出国家的直接差值。正负号也不能证明国家、类型或演员人数导致了电影时长的变化。

测试结果和讨论见 [分析报告](分析报告.md)。
