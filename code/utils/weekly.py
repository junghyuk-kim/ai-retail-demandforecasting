"""Daily to weekly aggregation for Ecuador retail data."""
from __future__ import annotations

import pandas as pd


def add_year_week(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    iso = out[date_col].dt.isocalendar()
    out["year"] = iso.year.astype(int)
    out["week"] = iso.week.astype(int)
    out["yearweek"] = out["year"] * 100 + out["week"]
    return out


def to_weekly_type_family(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily df to type x family x week (system-level panel for clustering)."""
    d = add_year_week(df)
    agg = (
        d.groupby(["type", "family", "year", "week", "yearweek"], as_index=False)
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
            transactions=("transactions", "sum"),
            dcoilwtico=("dcoilwtico", "mean"),
            is_holiday=("is_holiday", "max"),
        )
    )
    return agg.sort_values(["type", "family", "yearweek"]).reset_index(drop=True)
