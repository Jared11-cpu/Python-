"""Netflix 电影时长回归。运行 python analysis.py 重新生成全部结果。"""
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
NUM = ['release_year', 'cast_count', 'director_count', 'country_count', 'genre_count']
CAT = ['primary_country', 'primary_genre', 'rating']
ALPHAS = np.logspace(-2, 2, 13)
SEED = 42


def load_data(path=ROOT / 'data/netflix_titles.csv'):
    raw = pd.read_csv(path)
    movies = raw.loc[raw['type'].eq('Movie')].copy()
    movies['minutes'] = pd.to_numeric(movies['duration'].str.extract(r'^(\d+) min$')[0], errors='coerce')
    movies = movies.drop_duplicates('show_id')
    movies = movies.loc[movies['minutes'].gt(0)].copy()
    for source, target in [('cast', 'cast_count'), ('director', 'director_count'),
                           ('country', 'country_count'), ('listed_in', 'genre_count')]:
        movies[target] = movies[source].map(
            lambda s: len([v for v in s.split(',') if v.strip()]) if isinstance(s, str) else np.nan)
    movies['primary_country'] = movies['country'].str.split(',').str[0].str.strip()
    movies['primary_genre'] = movies['listed_in'].str.split(',').str[0].str.strip()
    return raw, movies


def split_data(movies):
    return train_test_split(movies[NUM + CAT], movies['minutes'], test_size=.2, random_state=SEED)


def pipeline(model):
    # 统计量只在当前训练折拟合；编码后所有列再做 Z-score。
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='Unknown')),
                          ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                                   min_frequency=20, sparse_output=False))]), CAT)
    ])
    return Pipeline([('features', pre), ('scale', StandardScaler()), ('model', model)])


def metrics(y, pred):
    return {'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAE': float(mean_absolute_error(y, pred)), 'R2': float(r2_score(y, pred))}


def run():
    warnings.filterwarnings('error', category=ConvergenceWarning)
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    raw, movies = load_data()
    Xtr, Xte, ytr, yte = split_data(movies)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    models = {'Ridge': (Ridge(), {'model__alpha': np.logspace(-2, 5, 22)}),
              'Lasso': (Lasso(max_iter=30000, tol=1e-5), {'model__alpha': ALPHAS}),
              'ElasticNet': (ElasticNet(max_iter=30000, tol=1e-5),
                             {'model__alpha': ALPHAS, 'model__l1_ratio': [.1, .5, .9]})}
    rows, fits, cv_tables = [], {}, []
    base = DummyRegressor().fit(Xtr, ytr)
    rows.append({'model': 'Mean baseline', **metrics(yte, base.predict(Xte)), 'nonzero': 0})
    selection = {}
    for name, (model, grid) in models.items():
        search = GridSearchCV(pipeline(model), grid, cv=cv, scoring='neg_mean_squared_error',
                              n_jobs=1, error_score='raise').fit(Xtr, ytr)
        table = pd.DataFrame(search.cv_results_)
        table['cv_mse'] = -table['mean_test_score']
        table['se_mse'] = table['std_test_score'] / np.sqrt(10)
        best = search.best_index_
        threshold = table.loc[best, 'cv_mse'] + table.loc[best, 'se_mse']
        eligible = table.loc[table['cv_mse'].le(threshold)]
        # 在 1-SE 范围内：优先非零系数最少，其次 alpha 更大，再次 CV MSE 更低。
        candidates = []
        for i, row in eligible.iterrows():
            fit = clone(search.estimator).set_params(**row['params']).fit(Xtr, ytr)
            nnz = int(np.count_nonzero(np.abs(fit['model'].coef_) > 1e-8))
            candidates.append((nnz, -row['params']['model__alpha'], row['cv_mse'], i, fit))
        selected = min(candidates, key=lambda c: c[:3])
        for label, fit, idx in [(name, search.best_estimator_, best),
                                (name + ' (1-SE)', selected[4], selected[3])]:
            fits[label] = fit
            rows.append({'model': label, **metrics(yte, fit.predict(Xte)),
                         'nonzero': int(np.count_nonzero(np.abs(fit['model'].coef_) > 1e-8)),
                         'alpha': float(fit['model'].alpha),
                         'l1_ratio': getattr(fit['model'], 'l1_ratio', np.nan),
                         'cv_RMSE': float(np.sqrt(table.loc[idx, 'cv_mse']))})
        selection[name] = {'best_params': search.best_params_, 'threshold_mse': float(threshold),
                           'one_se_params': table.loc[selected[3], 'params']}
        table['family'] = name
        cv_tables.append(table[['family', 'param_model__alpha', 'cv_mse', 'se_mse']].assign(
            l1_ratio=table.get('param_model__l1_ratio', np.nan)))
        print(name, search.best_params_, flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'metrics.csv', index=False)
    pd.concat(cv_tables).to_csv(out / 'cross_validation.csv', index=False)
    en = fits['ElasticNet']
    names = en['features'].get_feature_names_out()
    coef = pd.DataFrame({'feature': names, 'coefficient': en['model'].coef_})
    coef = coef.iloc[np.argsort(-np.abs(coef.coefficient))]
    coef.to_csv(out / 'coefficients.csv', index=False)
    pred = en.predict(Xte)
    pd.DataFrame({'show_id': movies.loc[Xte.index, 'show_id'], 'actual_minutes': yte,
                  'predicted_minutes': pred, 'residual': yte - pred}).to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame({'show_id': movies.loc[Xtr.index, 'show_id'], 'split': 'train'}).to_csv(out / 'train_ids.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(movies.minutes, bins=45, color='#0f766e', edgecolor='white')
    axes[0].set(xlabel='Movie duration (minutes)', ylabel='Count', title='Netflix catalogue: movies only')
    axes[1].scatter(yte, pred, s=10, alpha=.3, color='#0f766e')
    limits = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    axes[1].plot(limits, limits, '--', color='#b45309')
    axes[1].set(xlabel='Actual minutes', ylabel='Predicted minutes', title='Elastic Net: held-out test set')
    fig.tight_layout(); fig.savefig(out / 'diagnostics.svg'); plt.close(fig)
    # 路径只用训练集；固定 EN 的 l1_ratio，避免把不同目标函数混成一条路径。
    Z = en[:-1].transform(Xtr)
    path_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (model, grid)) in zip(axes, models.items()):
        values = []
        path_alphas = grid['model__alpha']
        for a in path_alphas:
            m = clone(model).set_params(alpha=a)
            if name == 'ElasticNet':
                m.set_params(l1_ratio=en['model'].l1_ratio)
            m.fit(Z, ytr)
            values.append(m.coef_)
            path_rows.extend({'model': name, 'alpha': a, 'feature': f, 'coefficient': c}
                             for f, c in zip(names, m.coef_))
        ax.semilogx(path_alphas, np.asarray(values))
        ax.set(title=name, xlabel='alpha (log scale)', ylabel='Coefficient (minutes / SD)')
    fig.tight_layout(); fig.savefig(out / 'coefficient_paths.svg'); plt.close(fig)
    pd.DataFrame(path_rows).to_csv(out / 'coefficient_paths.csv', index=False)
    missing = raw.isna().sum().to_dict()
    manifest = {'source': 'https://www.kaggle.com/datasets/shivamb/netflix-shows',
                'downloaded': '2026-11-25',
                'sha256': hashlib.sha256((ROOT / 'data/netflix_titles.csv').read_bytes()).hexdigest(),
                'raw_rows': len(raw), 'raw_movies': int(raw.type.eq('Movie').sum()),
                'valid_movies': len(movies), 'train_rows': len(Xtr), 'test_rows': len(Xte),
                'features_after_encoding': len(names), 'missing_raw': missing,
                'duration_summary': movies.minutes.describe().to_dict(), 'selection': selection,
                'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                             'numpy': np.__version__, 'scikit-learn': sklearn.__version__}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(scores.to_string(index=False), flush=True)
    from write_report import write_report
    write_report()
    return scores, manifest


if __name__ == '__main__':
    run()
