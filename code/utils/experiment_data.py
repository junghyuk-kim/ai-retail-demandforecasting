"""Shared data loading for forecasting experiment notebooks."""
from __future__ import annotations

import pandas as pd

from .paths import DATA_PROCESSED, SBC_CLUSTER, ML_CLUSTER


def load_forecast_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """주간 판매 + 피처 + SBC/ML 클러스터 라벨 병합."""
    df = pd.read_parquet(DATA_PROCESSED / "df_weekly.parquet")
    feat_path = DATA_PROCESSED / "df_weekly_features.parquet"
    feat_df = pd.read_parquet(feat_path) if feat_path.exists() else df.copy()
    sbc = pd.read_parquet(SBC_CLUSTER)
    ml = pd.read_parquet(ML_CLUSTER)

    for d in (df, feat_df):
        d.drop(
            columns=[c for c in d.columns if c in ("SBC_CLUSTER", "ML_CLUSTER")],
            errors="ignore",
            inplace=True,
        )

    cl_sbc = sbc[["type", "family", "SBC_CLUSTER"]]
    cl_ml = ml[["type", "family", "ML_CLUSTER"]]
    df = df.merge(cl_sbc, on=["type", "family"], how="left")
    df = df.merge(cl_ml, on=["type", "family"], how="left")
    feat_df = feat_df.merge(cl_sbc, on=["type", "family"], how="left")
    feat_df = feat_df.merge(cl_ml, on=["type", "family"], how="left")
    return df, feat_df
