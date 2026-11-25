"""运行结果和交互功能的检查；先运行 analysis.py。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from analysis import ROOT, NUM, CAT, load_data, split_data


def main():
    raw, movies = load_data()
    Xtr, Xte, ytr, yte = split_data(movies)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert len(Xtr) + len(Xte) == len(movies)
    assert 'duration' not in NUM + CAT and 'minutes' not in NUM + CAT
    assert (movies.minutes > 0).all() and movies.show_id.is_unique
    out = ROOT / 'results'
    info = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    scores = pd.read_csv(out / 'metrics.csv')
    assert len(scores) == 7
    assert np.isfinite(scores[['RMSE', 'MAE', 'R2']]).all().all()
    preds = pd.read_csv(out / 'predictions.csv')
    assert len(preds) == len(yte)
    np.testing.assert_allclose(preds.actual_minutes, yte)
    rmse = np.sqrt(np.mean((preds.actual_minutes - preds.predicted_minutes)**2))
    np.testing.assert_allclose(rmse, scores.loc[scores.model.eq('ElasticNet'), 'RMSE'].iloc[0])
    assert info['train_rows'] == len(Xtr) and info['test_rows'] == len(Xte)
    for name in ['Ridge', 'Lasso', 'ElasticNet']:
        s = info['selection'][name]
        cv = pd.read_csv(out / 'cross_validation.csv')
        selected = cv.loc[cv.family.eq(name) & np.isclose(cv.param_model__alpha, s['one_se_params']['model__alpha'])]
        if name == 'ElasticNet':
            selected = selected.loc[np.isclose(selected.l1_ratio, s['one_se_params']['model__l1_ratio'])]
        assert len(selected) == 1 and selected.cv_mse.iloc[0] <= s['threshold_mse'] + 1e-8
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    initial = app.metric[3].value
    app.slider[0].set_value(2.0).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    assert app.metric[3].value != initial, 'alpha 改变后验证指标应更新'
    app.slider[0].set_value(-.3).run(timeout=60)
    before_ratio = app.metric[3].value
    app.slider[1].set_value(.9).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    assert app.metric[3].value != before_ratio, 'l1_ratio 改变后验证指标应更新'
    assert (out / 'diagnostics.svg').stat().st_size > 1000
    assert (out / 'coefficient_paths.svg').stat().st_size > 1000
    print('PASS: data split, leakage exclusions, 7 model results, RMSE recalculation, 1-SE selection, app startup and both sliders.')


if __name__ == '__main__':
    main()
