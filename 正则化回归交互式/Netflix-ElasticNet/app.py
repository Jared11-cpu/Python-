"""运行：streamlit run app.py。交互试验只展示训练集内部验证结果。"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from analysis import ROOT, load_data, split_data, pipeline, metrics

st.set_page_config(page_title='正则化回归交互式', layout='wide')
st.title('正则化回归交互式')
st.caption('Netflix 电影时长分析 · Kaggle / shivamb/netflix-shows')
st.write('根据上映年份、主要国家、类型、分级及演职员数量，估计一部电影的时长。')
st.info('这里只分析电影，时长按分钟计算。rating 一列记录的是 PG-13、TV-MA 等内容分级。')

@st.cache_data
def data():
    return load_data()

raw, movies = data()
tab1, tab2, tab3 = st.tabs(['看看数据', '调节参数', '模型比较'])
with tab1:
    a, b, c = st.columns(3)
    a.metric('原始条目', len(raw))
    b.metric('有效电影', len(movies))
    c.metric('电影时长中位数', f'{movies.minutes.median():.0f} 分钟')
    st.bar_chart(pd.cut(movies.minutes, bins=np.arange(0, 341, 20)).value_counts(sort=False).rename_axis('分钟区间').astype(int).set_axis(
        [f'{i}–{i+20}' for i in range(0, 340, 20)]))
    st.dataframe(raw.head(30), hide_index=True)
    st.subheader('缺失值数量')
    st.dataframe(raw.isna().sum().rename('缺失数'), width='stretch')
with tab2:
    st.write('拖动滑块，看看误差和保留下来的系数有什么变化。这里用训练数据的 25% 做验证，测试集单独保留。')
    left, right = st.columns([1, 2])
    with left:
        exponent = st.slider('log10(alpha)：惩罚强度', -2.0, 2.0, -0.3, .1)
        ratio = st.slider('l1_ratio：L1 惩罚占比', .1, .9, .5, .1)
        st.caption('越靠近 1 越接近 Lasso；越靠近 0，L2 收缩的作用越强。')
        st.latex(r'\frac{\|y-X\beta\|_2^2}{2n}+\alpha\rho\|\beta\|_1+\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2')
    @st.cache_data
    def experiment(exponent, ratio):
        X, _, y, _ = split_data(movies)
        Xa, Xv, ya, yv = train_test_split(X, y, test_size=.25, random_state=17)
        fit = pipeline(ElasticNet(alpha=10**exponent, l1_ratio=ratio, max_iter=30000, tol=1e-5)).fit(Xa, ya)
        weights = pd.DataFrame({'feature': fit['features'].get_feature_names_out(), 'coefficient': fit['model'].coef_})
        return metrics(yv, fit.predict(Xv)), weights
    score, weights = experiment(exponent, ratio)
    with right:
        a, b, c = st.columns(3)
        a.metric('验证 RMSE', f"{score['RMSE']:.2f} 分钟")
        b.metric('验证 R²', f"{score['R2']:.3f}")
        c.metric('非零系数', int((weights.coefficient.abs() > 1e-8).sum()))
        st.caption(f'当前 alpha = {10**exponent:.4f}；系数表示编码并标准化后一单位标准差的变化。')
        top = weights.loc[weights.coefficient.abs().nlargest(15).index].set_index('feature')
        st.bar_chart(top)
        st.dataframe(top, width='stretch')
with tab3:
    path = ROOT / 'results/metrics.csv'
    if path.exists():
        scores = pd.read_csv(path)
        st.dataframe(scores.round(4), hide_index=True, width='stretch')
        st.caption('参数通过 10 折交叉验证选出。1-SE 版本允许验证误差稍高一点，换取更少的非零系数。')
        st.image(str(ROOT / 'results/diagnostics.svg'))
        st.image(str(ROOT / 'results/coefficient_paths.svg'))
        st.download_button('下载模型比较表', path.read_bytes(), 'metrics.csv', 'text/csv')
    else:
        st.warning('请先运行 python analysis.py 生成正式分析结果。')
    st.write('这些结果反映的是电影信息与时长的关联。想判断模型对新片是否同样有效，还需要按年份另做一次检验。')
