"""Weighted WMAPE aggregation and SBC vs ML scheme comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import VAL_WEEKS


def validation_weights(df: pd.DataFrame, val_weeks: list | None = None) -> pd.DataFrame:
    """검증 구간 절대 판매량 합 = 제품별 WMAPE 가중치."""
    weeks = val_weeks if val_weeks is not None else VAL_WEEKS
    val = df[df["yearweek"].isin(weeks)]
    w = (
        val.groupby(["type", "family"])["sales"]
        .apply(lambda s: np.abs(s.astype(float)).sum())
        .reset_index(name="val_weight")
    )
    return w


def weighted_wmape(values: pd.Series, weights: pd.Series) -> float:
    w = weights.sum()
    if w == 0:
        return np.nan
    return float((values * weights).sum() / w)


def _pick_detail(
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
    scheme: str,
    typ: str,
    cluster: int,
    winner: str,
    phase1_model: str,
    phase2_model: str,
) -> pd.DataFrame:
    if winner == "Phase2":
        mask = (
            (phase2["cluster_scheme"] == scheme)
            & (phase2["type"] == typ)
            & (phase2["cluster"] == cluster)
            & (phase2["model"] == phase2_model)
        )
        return phase2[mask]
    mask = (
        (phase1["cluster_scheme"] == scheme)
        & (phase1["type"] == typ)
        & (phase1["cluster"] == cluster)
        & (phase1["model"] == phase1_model)
    )
    return phase1[mask]


def build_family_from_phase2_best(
    phase2: pd.DataFrame,
    phase2_best: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """조건별 XGBoost+Best 임베딩 제품 단위 WMAPE."""
    parts = []
    for row in phase2_best.itertuples(index=False):
        detail = phase2[
            (phase2["cluster_scheme"] == row.cluster_scheme)
            & (phase2["type"] == row.type)
            & (phase2["cluster"] == row.cluster)
            & (phase2["model"] == row.best_hybrid)
        ]
        if detail.empty:
            continue
        detail = detail.merge(weights, on=["type", "family"], how="left")
        detail["final_model"] = row.best_hybrid
        parts.append(detail)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out[
        ["cluster_scheme", "type", "cluster", "family", "mape", "wmape", "val_weight", "final_model", "embedding"]
    ]


def thesis_wmape_by_type(family_df: pd.DataFrame) -> pd.DataFrame:
    """논문식 WMAPE = Σ_k n_k·MAPE_k / Σ_k n_k (클러스터 제품수 가중, §4.5 Eq.50).

    각 (type, scheme)에 대해 클러스터별 평균 MAPE를 제품수로 가중 평균.
    반환: type별 SBC vs ML WMAPE와 우세 scheme.
    """
    rows = []
    for (typ, scheme), g in family_df.groupby(["type", "cluster_scheme"]):
        by_cluster = g.groupby("cluster")["mape"].agg(["mean", "count"])
        n_total = by_cluster["count"].sum()
        weighted_sum = float((by_cluster["mean"] * by_cluster["count"]).sum())
        rows.append({
            "type": typ, "scheme": scheme,
            "n_products": int(n_total),
            "weighted_sum": round(weighted_sum, 2),
            "wmape": round(weighted_sum / n_total, 2) if n_total else np.nan,
        })
    long = pd.DataFrame(rows)
    piv = long.pivot(index="type", columns="scheme", values="wmape")
    out = pd.DataFrame(index=piv.index)
    out["SBC_wmape"] = piv.get("SBC")
    out["ML_wmape"] = piv.get("ML")
    out["delta_SBC_minus_ML"] = out["SBC_wmape"] - out["ML_wmape"]
    out["better_scheme"] = np.where(
        out["delta_SBC_minus_ML"] < 0, "SBC",
        np.where(out["delta_SBC_minus_ML"] > 0, "ML", "tie"),
    )
    return out.round(2)


def build_family_final_results(
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
    final_best: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """조건별 최종 모델의 제품(family) 단위 지표 + WMAPE 가중치."""
    parts = []
    for row in final_best.itertuples(index=False):
        p2_model = row.phase2_best
        detail = _pick_detail(
            phase1,
            phase2,
            row.cluster_scheme,
            row.type,
            int(row.cluster),
            row.winner,
            row.phase1_best,
            p2_model,
        )
        if detail.empty:
            continue
        detail = detail.merge(weights, on=["type", "family"], how="left")
        detail["final_model"] = row.final_model
        detail["winner"] = row.winner
        parts.append(detail)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out[
        [
            "cluster_scheme",
            "type",
            "cluster",
            "family",
            "wmape",
            "val_weight",
            "final_model",
            "winner",
        ]
    ]


def summarize_wmape(
    family_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """그룹별 단순평균 vs 판매량 가중 WMAPE + 제품 수."""
    rows = []
    for keys, g in family_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_products": len(g),
                "wmape_mean": g["wmape"].mean(),
                "wmape_weighted": weighted_wmape(g["wmape"], g["val_weight"]),
                "val_sales_sum": g["val_weight"].sum(),
            }
        )
    return pd.DataFrame(rows)


def compare_schemes_by_type(family_df: pd.DataFrame) -> pd.DataFrame:
    """type별 SBC(rule-base) vs ML 가중 WMAPE 비교."""
    by_type = summarize_wmape(family_df, ["type", "cluster_scheme"])
    pivot_w = by_type.pivot(index="type", columns="cluster_scheme", values="wmape_weighted")
    pivot_m = by_type.pivot(index="type", columns="cluster_scheme", values="wmape_mean")
    pivot_n = by_type.pivot(index="type", columns="cluster_scheme", values="n_products")

    out = pd.DataFrame(index=sorted(family_df["type"].unique()))
    out["n_products"] = pivot_n["SBC"]
    out["SBC_wmape_weighted"] = pivot_w["SBC"]
    out["ML_wmape_weighted"] = pivot_w["ML"]
    out["SBC_wmape_mean"] = pivot_m["SBC"]
    out["ML_wmape_mean"] = pivot_m["ML"]
    out["delta_weighted_SBC_minus_ML"] = out["SBC_wmape_weighted"] - out["ML_wmape_weighted"]
    out["better_scheme_weighted"] = np.where(
        out["delta_weighted_SBC_minus_ML"] < 0,
        "SBC",
        np.where(out["delta_weighted_SBC_minus_ML"] > 0, "ML", "tie"),
    )
    return out.round(2)


def compare_schemes_by_cluster(family_df: pd.DataFrame) -> pd.DataFrame:
    """type×cluster 조건별 SBC vs ML (동일 type 내 클러스터 번호는 별개 축)."""
    sbc = summarize_wmape(family_df[family_df["cluster_scheme"] == "SBC"], ["type", "cluster"])
    ml = summarize_wmape(family_df[family_df["cluster_scheme"] == "ML"], ["type", "cluster"])
    sbc = sbc.rename(
        columns={
            "wmape_mean": "SBC_wmape_mean",
            "wmape_weighted": "SBC_wmape_weighted",
            "n_products": "SBC_n",
        }
    )
    ml = ml.rename(
        columns={
            "wmape_mean": "ML_wmape_mean",
            "wmape_weighted": "ML_wmape_weighted",
            "n_products": "ML_n",
        }
    )
    merged = sbc.merge(ml, on=["type", "cluster"], how="outer")
    merged["better_weighted"] = np.where(
        merged["SBC_wmape_weighted"] < merged["ML_wmape_weighted"],
        "SBC",
        np.where(merged["SBC_wmape_weighted"] > merged["ML_wmape_weighted"], "ML", "tie"),
    )
    return merged.round(2)
