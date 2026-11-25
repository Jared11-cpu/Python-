"""Run the Iris clustering experiment and save its figures and tables."""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def save_figures(result):
    best_k = result["bestK"]
    raw = result["kmeans"]["4d"][best_k - 2]
    reduced = result["kmeans"]["2d"][best_k - 2]
    fig, (ax, blade) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    ax.plot(range(1, 9), result["elbow"], marker="o", color="#185c4e")
    ax.axvline(result["elbowK"], color="#a8492e", linestyle="--")
    ax.set(xlabel="K", ylabel="WCSS", title="Elbow curve", xticks=range(1, 9))
    start = 0
    for cluster, scores in enumerate(raw["blades"]):
        rows = np.arange(start, start + len(scores))
        blade.barh(rows, scores, height=1, label=f"Cluster {cluster + 1}")
        start += len(scores) + 5
    blade.axvline(raw["silhouette"], color="#333333", linestyle="--")
    blade.set(xlabel="Silhouette coefficient", title=f"Silhouette blades (K={best_k})",
              xlim=(-1, 1), yticks=[])
    blade.legend(loc="upper left")
    fig.savefig(HERE / "kmeans_diagnostics.png", dpi=150)
    plt.close(fig)

    index = result["defaultQuantile"]
    db = result["dbscan"]
    pairs = [
        (result["xy"], raw["labels"], reduced["labels"], f"K-Means (K={best_k})"),
        (result["xy"], db["4d"]["sweep"][index]["labels"],
         db["2d"]["sweep"][index]["labels"], "DBSCAN: Iris"),
        (result["noisyXY"], db["104d"]["sweep"][index]["labels"],
         db["noisy2d"]["sweep"][index]["labels"], "DBSCAN: Iris + noise"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9, 10), sharex="row", sharey="row", layout="constrained")
    for row, (xy, left, right, model) in zip(axes, pairs):
        xy = np.asarray(xy)
        for ax, labels, fit in zip(row, (left, right), ("Full-space fit", "PCA 2D fit")):
            labels = np.asarray(labels)
            for cluster in np.unique(labels):
                points = xy[labels == cluster]
                if cluster == -1:
                    ax.scatter(*points.T, s=22, color="#6d756d", marker="x", label="Noise")
                else:
                    ax.scatter(*points.T, s=18, label=f"Cluster {cluster + 1}")
            ax.set(title=f"{model} / {fit}", xlabel="PC1", ylabel="PC2")
            ax.legend(fontsize=8)
    fig.savefig(HERE / "pca_comparison.png", dpi=150)
    plt.close(fig)


def save_tables(result):
    lines = [
        "# Iris 实验结果", "",
        f"手肘候选 K = {result['elbowK']}，平均轮廓系数最高的 K = {result['bestK']}。", "",
        "| K | 4 维 WCSS | 4 维平均轮廓 | 4 维 ARI | PCA 2 维 ARI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for raw, reduced in zip(result["kmeans"]["4d"], result["kmeans"]["2d"]):
        lines.append(f"| {raw['k']} | {raw['wcss']:.2f} | {raw['silhouette']:.4f} "
                     f"| {raw['ari']:.4f} | {reduced['ari']:.4f} |")
    lines.extend([
        "", "DBSCAN 取各空间第 5 近邻距离的第 90 百分位为 ε。", "",
        "| 拟合空间 | ε | 簇数 | 噪声点数 | ARI | 相对距离跨度 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    names = {"4d": "Iris 4 维", "2d": "Iris PCA 2 维",
             "104d": "加噪 104 维", "noisy2d": "加噪 PCA 2 维"}
    for key, space in result["dbscan"].items():
        row = space["sweep"][result["defaultQuantile"]]
        lines.append(f"| {names[key]} | {row['eps']:.5f} | {row['clusters']} | {row['noise']} "
                     f"| {row['ari']:.4f} | {space['spread']:.4f} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    save_figures(result)
    save_tables(result)


if __name__ == "__main__":
    main()
