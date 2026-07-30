"""Embedding x clustering grid search (6 embedders x 4 clusterers)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .embeddings import EMBEDDERS
from .metrics import clustering_quality


CLUSTERERS = ("KMeans", "HAC", "GMM", "DBSCAN")


def merge_small_clusters(
    X_emb: np.ndarray, labels: np.ndarray, min_size: int = 3
) -> np.ndarray:
    """min_size 미만 클러스터를 가장 가까운(centroid 거리) 생존 클러스터로 병합.

    학위논문 원본의 outlier 재클러스터링 단계에 대응. 축소 데이터(2-type)에서
    임베딩 클러스터가 '거대 클러스터 1 + 크기 1~2 outlier'로 퇴화하는 것을 방지한다.
    반환 라벨은 0-based 연속값으로 재부여.
    """
    labels = np.asarray(labels).copy()
    while True:
        counts = Counter(labels[labels >= 0].tolist())
        small = [c for c, n in counts.items() if n < min_size]
        big = [c for c, n in counts.items() if n >= min_size]
        if not small or not big:
            break
        cents = {c: X_emb[labels == c].mean(axis=0) for c in big}
        moved = False
        for c in small:
            for i in np.where(labels == c)[0]:
                nearest = min(big, key=lambda bc: np.linalg.norm(X_emb[i] - cents[bc]))
                labels[i] = nearest
                moved = True
        if not moved:
            break
    # 연속 0-based 재라벨
    uniq = {c: i for i, c in enumerate(sorted(set(labels[labels >= 0].tolist())))}
    return np.array([uniq.get(l, -1) for l in labels])


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
        # reg_covar 상향: 소표본(2-type 66개)에서 특이/붕괴 공분산 방지
        return GaussianMixture(
            n_components=k, random_state=42, n_init=10, reg_covar=1e-3
        ).fit_predict(Xs)
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
            try:
                labels = fit_cluster_labels(X_emb, cl_name, k=k)
                q = clustering_quality(X_emb, labels)
            except Exception as exc:  # 소표본에서 특정 조합 수치 실패 → NaN 기록 후 계속
                labels = np.zeros(X_emb.shape[0], dtype=int)
                q = {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": 0}
                print(f"  {combo}: SKIPPED ({type(exc).__name__})")
            counts = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([0])
            q["min_cluster_size"] = int(counts.min()) if counts.size else 0
            q.update({"embedding": emb_name, "clustering": cl_name, "method": combo})
            rows.append(q)
            label_cache[(emb_name, cl_name)] = labels
            print(
                f"  {combo}: silhouette={q['silhouette']}, "
                f"db={q['davies_bouldin']}, n_clusters={q['n_clusters']}"
            )

    quality_df = pd.DataFrame(rows)[
        ["method", "embedding", "clustering", "n_clusters", "min_cluster_size",
         "silhouette", "davies_bouldin"]
    ]
    return quality_df, label_cache


def select_best_combo(quality_df: pd.DataFrame, k: int = 4, balance_floor: int = 4) -> pd.Series:
    """Pick best combo by silhouette (high) + Davies-Bouldin (low), with a balance guard.

    balance_floor: 최소 클러스터 크기가 이 값 이상인 조합만 후보로 삼아, outlier 1개만
    떼어낸 degenerate 해(높은 실루엣이지만 forecastable하지 않음)를 배격한다(논문 방식).
    충족 조합이 없으면 floor를 완화한다.
    """
    valid = quality_df[
        quality_df["silhouette"].notna() & quality_df["davies_bouldin"].notna()
    ].copy()
    if valid.empty:
        return quality_df.iloc[0]

    # Prefer fixed-K methods (KMeans/HAC/GMM) matching the SBC cluster count.
    fixed_k = valid[valid["n_clusters"] == k]
    if not fixed_k.empty:
        valid = fixed_k

    # Balance guard: 최소 클러스터 크기 >= floor 조합 우선 (없으면 완화)
    if "min_cluster_size" in valid.columns:
        for floor in range(balance_floor, 1, -1):
            balanced = valid[valid["min_cluster_size"] >= floor]
            if not balanced.empty:
                valid = balanced
                break

    valid["sil_rank"] = valid["silhouette"].rank(ascending=False)
    valid["db_rank"] = valid["davies_bouldin"].rank(ascending=True)
    valid["rank_score"] = valid["sil_rank"] + valid["db_rank"]
    cluster_order = {"KMeans": 0, "HAC": 1, "GMM": 2, "DBSCAN": 3}
    valid["cl_ord"] = valid["clustering"].map(cluster_order).fillna(9)
    best = valid.sort_values(
        ["rank_score", "davies_bouldin", "silhouette", "cl_ord"],
        ascending=[True, True, False, True],
    ).iloc[0]
    return best


def add_joint_rank(quality_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Return quality table with joint silhouette+DB rank columns."""
    out = quality_df.copy()
    fixed = out[out["n_clusters"] == k].copy()
    if fixed.empty:
        fixed = out[out["n_clusters"] >= 2].copy()
    fixed["sil_rank"] = fixed["silhouette"].rank(ascending=False)
    fixed["db_rank"] = fixed["davies_bouldin"].rank(ascending=True)
    fixed["rank_score"] = fixed["sil_rank"] + fixed["db_rank"]
    return fixed.sort_values(["rank_score", "davies_bouldin"]).reset_index(drop=True)
