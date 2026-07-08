"""Embedding x clustering grid search (6 embedders x 4 clusterers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .embeddings import EMBEDDERS
from .metrics import clustering_quality


CLUSTERERS = ("KMeans", "HAC", "GMM", "DBSCAN")


def _dbscan_eps(X: np.ndarray, min_samples: int = 3) -> float:
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    return float(np.median(dists[:, -1]))


def fit_cluster_labels(
    X_emb: np.ndarray,
    method: str,
    k: int = 4,
    dbscan_min_samples: int = 3,
) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X_emb)
    if method == "KMeans":
        return KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xs)
    if method == "HAC":
        return AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
    if method == "GMM":
        return GaussianMixture(n_components=k, random_state=42, n_init=10).fit_predict(Xs)
    if method == "DBSCAN":
        eps = _dbscan_eps(Xs, min_samples=dbscan_min_samples)
        return DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit_predict(Xs)
    raise ValueError(f"Unknown clustering method: {method}")


def run_embedding_clustering_grid(
    X_raw: np.ndarray,
    k: int = 4,
    n_components: int = 10,
    embedders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run all embedding x clustering combinations; return quality table and label cache."""
    embedders = embedders or EMBEDDERS
    rows: list[dict] = []
    label_cache: dict[tuple[str, str], np.ndarray] = {}

    for emb_name, emb_fn in embedders.items():
        print(f"[embedding] {emb_name}")
        X_emb = emb_fn(X_raw, n_components=n_components)
        for cl_name in CLUSTERERS:
            combo = f"{emb_name}+{cl_name}"
            labels = fit_cluster_labels(X_emb, cl_name, k=k)
            q = clustering_quality(X_emb, labels)
            q.update({"embedding": emb_name, "clustering": cl_name, "method": combo})
            rows.append(q)
            label_cache[(emb_name, cl_name)] = labels
            print(
                f"  {combo}: silhouette={q['silhouette']}, "
                f"db={q['davies_bouldin']}, n_clusters={q['n_clusters']}"
            )

    quality_df = pd.DataFrame(rows)[
        ["method", "embedding", "clustering", "n_clusters", "silhouette", "davies_bouldin"]
    ]
    return quality_df, label_cache


def select_best_combo(quality_df: pd.DataFrame) -> pd.Series:
    valid = quality_df[quality_df["n_clusters"] >= 2].copy()
    if valid.empty:
        return quality_df.iloc[0]
    valid["rank_score"] = valid["silhouette"].rank(ascending=False) + valid[
        "davies_bouldin"
    ].rank(ascending=True)
    return valid.sort_values("rank_score").iloc[0]
