"""System-level and SKU-level (series-level) basic statistics by type."""
from __future__ import annotations

import numpy as np
import pandas as pd


def system_level_weekly_stats(df_weekly: pd.DataFrame, type_val: str) -> pd.DataFrame:
    """Type-level: sum sales across all families per week (system-level time series)."""
    sub = df_weekly[df_weekly["type"] == type_val]
    sys_ts = sub.groupby("yearweek", as_index=False)["sales"].sum()
    s = sys_ts["sales"]
    return pd.DataFrame(
        [{
            "type": type_val,
            "level": "system",
            "n_weeks": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=1),
            "variance": s.var(ddof=1),
            "cv": s.std(ddof=1) / s.mean() if s.mean() else np.nan,
            "min": s.min(),
            "max": s.max(),
        }]
    )


def sku_level_weekly_stats(df_weekly: pd.DataFrame, type_val: str) -> pd.DataFrame:
    """Type-level: per (type, family) series stats, then summarize across families."""
    sub = df_weekly[df_weekly["type"] == type_val]
    per_series = (
        sub.groupby(["type", "family"])["sales"]
        .agg(mean="mean", std="std", median="median", n_weeks="count")
        .reset_index()
    )
    per_series["cv"] = per_series["std"] / per_series["mean"].replace(0, np.nan)
    summary = {
        "type": type_val,
        "level": "sku",
        "n_series": per_series["family"].nunique(),
        "mean_of_mean": per_series["mean"].mean(),
        "mean_of_std": per_series["std"].mean(),
        "mean_of_cv": per_series["cv"].mean(),
        "median_of_mean": per_series["mean"].median(),
    }
    return pd.DataFrame([summary]), per_series
