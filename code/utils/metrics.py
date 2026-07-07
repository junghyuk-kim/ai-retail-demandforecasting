"""Forecasting and clustering evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def clustering_quality(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette (higher better) and Davies-Bouldin (lower better)."""
    labels = np.asarray(labels)
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2 or len(embeddings) < len(unique) + 1:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": len(unique)}
    mask = labels >= 0
    X = embeddings[mask]
    y = labels[mask]
    if len(np.unique(y)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": len(np.unique(y))}
    return {
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "n_clusters": int(len(np.unique(y))),
    }
