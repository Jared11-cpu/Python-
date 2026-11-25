# 正则化回归交互式

下载 [index.html](index.html)，用浏览器打开即可。页面包含第 1 题的 Ridge 推导、第 3 题的标准化分析，以及第 4 题的 Hitters 实验。实验可以切换模型、拖动 α、查看单个变量的系数，并比较最小误差与 1-SE 两种选参结果。

## 复现实验

在本目录运行：

```sh
python -m pip install -r requirements.txt
python analysis.py
```

脚本读取同目录的 [Hitters.csv](Hitters.csv)，打印测试集结果，并重新生成 `index.html`。`template.html` 是生成页面用的模板。数据取自 [ISLR 的 Hitters 数据](https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Hitters.html)，薪资单位为千美元。
