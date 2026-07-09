"""System-level and SKU-level (series-level) basic statistics by type."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, mannwhitneyu, skew

from .sbc import compute_sbc_from_series


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
            "skewness": skew(s, bias=False) if len(s) > 2 else np.nan,
            "kurtosis": kurtosis(s, fisher=True, bias=False) if len(s) > 3 else np.nan,
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


# ---------------------------------------------------------------------------
# Type-level demand variability profile & high/low-variation type selection
#
# 논문 3.3절: 고/저변동 센터 판별의 핵심 지표는 **System-Level CV**이며 (SKU-Level은
# 센터 간 거의 동일), 이 변동성 지표(CV/CV²)로 예측 전략(ML vs SBC)을 선택한다.
# 본 실습에서는 store type을 논문의 "센터"에 대응시켜, System-Level CV가 최대인
# type을 고변동, 최소인 type을 저변동으로 선정한다.
# ---------------------------------------------------------------------------


def _family_variability(sales: pd.Series) -> dict:
    """단일 (type, family) 시계열의 CV·CV²·ADI·zero_ratio·skew."""
    s = sales.astype(float)
    vals = s.values
    mean = vals.mean()
    sbc = compute_sbc_from_series(s)  # ADI, CV2 (비영 수요 기준, ddof=1)
    return {
        "cv": vals.std(ddof=1) / mean if mean else np.nan,
        "cv2": sbc["CV2"],
        "adi": sbc["ADI"],
        "zero_ratio": 1.0 - (sbc["n_positive"] / sbc["n_periods"]),
        "skew": skew(vals, bias=False) if len(vals) > 2 else np.nan,
    }


def type_variation_table(df_weekly: pd.DataFrame, week_max: int | None = None) -> pd.DataFrame:
    """type별 System-Level + SKU-Level 수요 변동성 프로파일 (논문 Table 3.2 스타일).

    week_max 지정 시 해당 주차 이하(학습 구간)만 사용. 반환 index = type.
    """
    d = df_weekly if week_max is None else df_weekly[df_weekly["yearweek"] <= week_max]
    rows = []
    for t in sorted(d["type"].unique()):
        sub = d[d["type"] == t]
        # --- System-Level: family 합산 주간 시계열 1개 ---
        sys_ts = sub.groupby("yearweek")["sales"].sum().sort_index().astype(float)
        s = sys_ts.values
        sys_mean = s.mean()
        # --- SKU-Level: family별 변동성 → 평균 ---
        fam_stats = pd.DataFrame(
            [_family_variability(g["sales"]) for _, g in sub.groupby("family")]
        )
        rows.append(
            {
                "type": t,
                "n_family": sub["family"].nunique(),
                "sys_mean": sys_mean,
                "sys_std": s.std(ddof=1),
                "sys_variance": s.var(ddof=1),
                "sys_cv": s.std(ddof=1) / sys_mean if sys_mean else np.nan,
                "sys_skew": skew(s, bias=False) if len(s) > 2 else np.nan,
                "sys_kurt": kurtosis(s, fisher=True, bias=False) if len(s) > 3 else np.nan,
                "sku_mean_cv": fam_stats["cv"].mean(),
                "sku_mean_cv2": fam_stats["cv2"].mean(),
                "sku_mean_adi": fam_stats["adi"].mean(),
                "sku_mean_skew": fam_stats["skew"].mean(),
                "sku_mean_zero_ratio": fam_stats["zero_ratio"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("type").sort_values("sys_cv", ascending=False)


def select_extreme_types(table: pd.DataFrame, metric: str = "sys_cv") -> dict:
    """변동성 지표(기본 System-Level CV) 최대=고변동, 최소=저변동 type 선정."""
    high = table[metric].idxmax()
    low = table[metric].idxmin()
    return {
        "high": str(high),
        "low": str(low),
        "metric": metric,
        "high_value": float(table.loc[high, metric]),
        "low_value": float(table.loc[low, metric]),
    }


def variation_significance_test(
    df_weekly: pd.DataFrame, type_a: str, type_b: str, week_max: int | None = None
) -> dict:
    """두 type의 family-level CV 분포 차이 검정 (논문 3.3절 Mann-Whitney U 대응)."""
    d = df_weekly if week_max is None else df_weekly[df_weekly["yearweek"] <= week_max]

    def _cvs(t: str) -> np.ndarray:
        sub = d[d["type"] == t]
        out = []
        for _, g in sub.groupby("family"):
            v = g["sales"].astype(float).values
            m = v.mean()
            if m:
                out.append(v.std(ddof=1) / m)
        return np.asarray(out, dtype=float)

    a, b = _cvs(type_a), _cvs(type_b)
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "type_a": type_a,
        "type_b": type_b,
        "median_cv_a": float(np.nanmedian(a)),
        "median_cv_b": float(np.nanmedian(b)),
        "mannwhitney_u": float(stat),
        "p_value": float(p),
    }
