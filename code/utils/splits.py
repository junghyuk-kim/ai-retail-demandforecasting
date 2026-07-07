"""Train/validation split helpers for weekly panels."""
from __future__ import annotations

import pandas as pd

# 검증: train 마지막 이후 3주 (2017-08-01~15 일별 구간에 대응하는 주간)
VAL_WEEKS = [201731, 201732, 201733]
TRAIN_WEEK_MAX = 201730


def add_split_flag(df: pd.DataFrame, week_col: str = "yearweek") -> pd.DataFrame:
    out = df.copy()
    out["split"] = out[week_col].apply(lambda w: "val" if w in VAL_WEEKS else "train")
    return out


def train_val_mask(df: pd.DataFrame, week_col: str = "yearweek") -> tuple[pd.Series, pd.Series]:
    train_m = df[week_col] <= TRAIN_WEEK_MAX
    val_m = df[week_col].isin(VAL_WEEKS)
    return train_m, val_m
