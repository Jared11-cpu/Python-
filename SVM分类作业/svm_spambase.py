"""Compare linear and RBF SVM classifiers on UCI Spambase."""
import argparse
import io
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
SEED = 42


def load_data(archive):
    if archive is None:
        with urlopen(DATA_URL, timeout=30) as response:
            archive = io.BytesIO(response.read())
    with ZipFile(archive) as dataset:
        with dataset.open("spambase.data") as csv:
            data = np.loadtxt(csv, delimiter=",")
    features, labels = data[:, :-1], data[:, -1].astype(int)
    # Keep each feature vector once and discard vectors with conflicting labels.
    _, first, groups = np.unique(
        features, axis=0, return_index=True, return_inverse=True
    )
    keep = np.bincount(groups, weights=labels) == (
        np.bincount(groups) * labels[first]
    )
    clean = np.sort(first[keep])
    return features[clean], labels[clean], len(data), int((~keep).sum())


def evaluate(name, model, x_test, y_test):
    predicted = model.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    print(
        f"{name}: accuracy={accuracy_score(y_test, predicted):.4f}, "
        f"precision={precision_score(y_test, predicted, zero_division=0):.4f}, "
        f"recall={recall_score(y_test, predicted, zero_division=0):.4f}, "
        f"F1={f1_score(y_test, predicted, zero_division=0):.4f}, "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local UCI ZIP instead of downloading")
    args = parser.parse_args()

    x, y, raw_count, conflicts = load_data(args.archive)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"scikit-learn {sklearn.__version__}; NumPy {np.__version__}")
    print(f"raw samples={raw_count}, conflicting feature groups={conflicts}")
    print(f"samples={len(y)}, features={x.shape[1]}, class counts={np.bincount(y).tolist()}")
    print(f"train={np.bincount(y_train).tolist()}, test={np.bincount(y_test).tolist()}")

    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    evaluate("Majority baseline", baseline, x_test, y_test)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for kernel, grid in [
        ("linear", {"svc__C": [0.1, 1, 10]}),
        ("rbf", {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, "scale"]}),
    ]:
        model = make_pipeline(StandardScaler(), SVC(kernel=kernel))
        search = GridSearchCV(model, grid, cv=folds, scoring="f1", error_score="raise")
        start = perf_counter()
        search.fit(x_train, y_train)
        elapsed = perf_counter() - start
        best = search.best_index_
        cv_std = search.cv_results_["std_test_score"][best]
        print(f"{kernel}: {search.best_params_}, CV F1={search.best_score_:.4f} "
              f"(std={cv_std:.4f}), search and refit={elapsed:.2f}s")
        evaluate(kernel, search, x_test, y_test)


if __name__ == "__main__":
    main()
