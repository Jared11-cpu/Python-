"""Generate the assignment report from the Iris data shipped with scikit-learn."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SEED = 42
MIN_SAMPLES = 5
QUANTILES = np.array([50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 99])


def dbscan_sweep(x, species):
    # An explicit query with x includes the point itself, as DBSCAN does.
    distances, _ = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(x).kneighbors(x)
    sweep = []
    for eps in np.percentile(distances[:, -1], QUANTILES):
        labels = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(x)
        sweep.append({
            "eps": round(float(eps), 5),
            "clusters": len(set(labels) - {-1}),
            "noise": int(np.count_nonzero(labels == -1)),
            "ari": round(adjusted_rand_score(species, labels), 5),
            "labels": labels.tolist(),
        })
    p10, median, p90 = np.percentile(pdist(x), [10, 50, 90])
    return {"sweep": sweep, "spread": round(float((p90 - p10) / median), 5)}


def kmeans_sweep(x, species):
    runs = []
    for k in range(2, 9):
        fitted = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(x)
        labels = fitted.labels_
        scores = silhouette_samples(x, labels)
        runs.append({
            "k": k, "wcss": round(fitted.inertia_, 5),
            "silhouette": round(float(scores.mean()), 5),
            "ari": round(adjusted_rand_score(species, labels), 5),
            "labels": labels.tolist(),
            "blades": [np.sort(scores[labels == i]).round(5).tolist() for i in range(k)],
        })
    return runs


def elbow_k(points):
    """Find the largest downward gap from the chord of the normalized curve."""
    values = np.asarray(points)
    scaled = (values - values[-1]) / (values[0] - values[-1])
    return int(np.argmax(np.linspace(1, 0, len(values)) - scaled) + 1)


def main():
    iris = load_iris()
    species = iris.target
    x = StandardScaler().fit_transform(iris.data)
    noisy = StandardScaler().fit_transform(np.column_stack(
        [iris.data, np.random.default_rng(SEED).normal(size=(len(x), 100))]
    ))
    pca = PCA(n_components=2, svd_solver="full")
    xy = pca.fit_transform(x)
    variance = pca.explained_variance_ratio_.round(5).tolist()
    noisy_xy = pca.fit_transform(noisy)
    noisy_variance = pca.explained_variance_ratio_.round(5).tolist()

    with threadpool_limits(limits=1):
        raw_runs = kmeans_sweep(x, species)
        pca_runs = kmeans_sweep(xy, species)
        first_wcss = float(np.square(x - x.mean(axis=0)).sum())
        elbow = [first_wcss] + [run["wcss"] for run in raw_runs]
        result = {
            "kmeans": {"4d": raw_runs, "2d": pca_runs},
            "elbow": elbow,
            "elbowK": elbow_k(elbow),
            "bestK": max(raw_runs, key=lambda run: run["silhouette"])["k"],
            "quantiles": QUANTILES.tolist(),
            "defaultQuantile": QUANTILES.tolist().index(90),
            "xy": xy.round(5).tolist(),
            "noisyXY": noisy_xy.round(5).tolist(),
            "variance": variance,
            "noisyVariance": noisy_variance,
            "dbscan": {name: dbscan_sweep(values, species) for name, values in
                       (("4d", x), ("2d", xy), ("104d", noisy), ("noisy2d", noisy_xy))},
        }

    print(f"Elbow K={result['elbowK']}; maximum silhouette K={result['bestK']}")
    for a, b in zip(raw_runs, pca_runs):
        print(f"K={a['k']} WCSS={a['wcss']:.2f} silhouette={a['silhouette']:.4f} "
              f"ARI 4D={a['ari']:.4f}, 2D={b['ari']:.4f}")
    for name, space in result["dbscan"].items():
        row = space["sweep"][result["defaultQuantile"]]
        print(f"DBSCAN {name}: eps={row['eps']:.5f} clusters={row['clusters']} "
              f"noise={row['noise']} ARI={row['ari']:.4f} spread={space['spread']:.4f}")
    print(f"PCA variance: Iris {sum(variance):.2%}, noise experiment {sum(noisy_variance):.2%}")
    template = (HERE / "template.html").read_text(encoding="utf-8")
    (HERE / "index.html").write_text(
        template.replace("__RESULTS__", json.dumps(result, separators=(",", ":"), allow_nan=False)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
