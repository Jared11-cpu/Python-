"""Fit Hitters salary models and build the offline interactive report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ALPHAS = np.logspace(-2, 3.5, 65)
TOL = 1e-8


def evaluate(model, reference, train_x, test_x, train_y, test_y, cv):
    scaler = StandardScaler().fit(train_x)
    x = scaler.transform(train_x)
    test = scaler.transform(test_x)
    # The built-in CV estimators see the whole training set's scale.
    reference.fit(x, train_y)

    # Refit the scale inside each fold for the reported model selection.
    folds = []
    for fit, val in cv.split(train_x):
        fold_scaler = StandardScaler().fit(train_x[fit])
        folds.append((fold_scaler.transform(train_x[fit]), train_y[fit],
                      fold_scaler.transform(train_x[val]), train_y[val]))

    paths, errors = [], []
    for alpha in ALPHAS:
        estimator = clone(model).set_params(alpha=alpha)
        estimator.fit(x, train_y)
        paths.append(estimator.coef_.tolist())
        errors.append([
            mean_squared_error(val_y, estimator.fit(fit_x, fit_y).predict(val_x)) ** 0.5
            for fit_x, fit_y, val_x, val_y in folds
        ])

    errors = np.array(errors)
    means = errors.mean(axis=1)
    se = errors.std(axis=1, ddof=1) / np.sqrt(cv.get_n_splits())
    best = int(means.argmin())
    one_se = int(np.flatnonzero(means <= means[best] + se[best])[-1])

    def summary(index):
        fitted = clone(model).set_params(alpha=ALPHAS[index]).fit(x, train_y)
        return {
            "index": index,
            "test": mean_squared_error(test_y, fitted.predict(test)) ** 0.5,
            "nonzero": int(np.count_nonzero(np.abs(fitted.coef_) > TOL)),
        }

    return {
        "baseline": {
            "alpha": float(reference.alpha_),
            "test": mean_squared_error(test_y, reference.predict(test)) ** 0.5,
            "nonzero": int(np.count_nonzero(np.abs(reference.coef_) > TOL)),
        },
        "coefficients": paths,
        "mean": means.tolist(),
        "se": se.tolist(),
        "min": summary(best),
        "one": summary(one_se),
    }


def main():
    data = pd.read_csv(HERE / "Hitters.csv").dropna(subset=["Salary"])
    features = pd.get_dummies(data.drop(columns=["Salary", "rownames"]),
                              columns=["League", "Division", "NewLeague"],
                              drop_first=True, dtype=float)
    train_x, test_x, train_y, test_y = train_test_split(
        features.to_numpy(), data["Salary"].to_numpy(), test_size=0.2, random_state=42
    )
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    models = {
        "Ridge": (Ridge(), RidgeCV(alphas=ALPHAS, cv=cv, scoring="neg_root_mean_squared_error")),
        "Lasso": (Lasso(max_iter=50000), LassoCV(alphas=ALPHAS, cv=cv, max_iter=50000)),
        "Elastic Net": (ElasticNet(l1_ratio=0.5, max_iter=50000),
                        ElasticNetCV(l1_ratio=0.5, alphas=ALPHAS, cv=cv, max_iter=50000)),
    }
    results = {"features": features.columns.tolist(), "alphas": ALPHAS.tolist(), "models": {}}
    print(f"Training: {len(train_y)}, test: {len(test_y)}, features: {features.shape[1]}")
    for name, (model, reference) in models.items():
        result = evaluate(model, reference, train_x, test_x, train_y, test_y, cv)
        results["models"][name] = result
        for rule in ("min", "one"):
            row = result[rule]
            i = row["index"]
            print(f"{name:12} {rule:3} alpha={ALPHAS[i]:8.3f} "
                  f"CV={result['mean'][i]:.2f} RMSE={row['test']:.2f} p={row['nonzero']}")
        baseline = result["baseline"]
        print(f"  sklearn CV alpha={baseline['alpha']:.3f} "
              f"RMSE={baseline['test']:.2f} p={baseline['nonzero']}")

    template = (HERE / "template.html").read_text(encoding="utf-8")
    (HERE / "index.html").write_text(
        template.replace("__RESULTS__", json.dumps(results, allow_nan=False)), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
