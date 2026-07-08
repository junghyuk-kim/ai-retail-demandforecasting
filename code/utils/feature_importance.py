"""XGBoost feature importance by store type."""
from __future__ import annotations

import pandas as pd

from .splits import TRAIN_WEEK_MAX


def feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"sales", "type", "family", "year", "week", "yearweek", "split"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _xgb_regressor():
    from xgboost import XGBRegressor

    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    try:
        import torch

        if torch.cuda.is_available():
            params["tree_method"] = "hist"
            params["device"] = "cuda"
    except Exception:
        pass
    return XGBRegressor(**params)


def fit_xgb_importance(
    train_df: pd.DataFrame,
    feature_cols_list: list[str],
) -> tuple[object, pd.Series]:
    """type 단위 학습 데이터로 XGBoost 적합 후 gain 기반 중요도 반환."""
    X = train_df[feature_cols_list].fillna(0)
    y = train_df["sales"].astype(float)
    model = _xgb_regressor()
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=feature_cols_list)
    imp = imp.sort_values(ascending=False)
    imp = (imp / imp.sum() * 100).round(4)  # 비율(%)
    return model, imp


def importance_by_type(
    feat_df: pd.DataFrame,
    types: list | None = None,
) -> pd.DataFrame:
    """5 type 각각 XGBoost 학습 → 변수 중요도 long format."""
    types = types or sorted(feat_df["type"].dropna().unique())
    cols = feature_cols(feat_df)
    rows = []
    for typ in types:
        train = feat_df[(feat_df["type"] == typ) & (feat_df["yearweek"] <= TRAIN_WEEK_MAX)]
        if train.empty:
            continue
        _, imp = fit_xgb_importance(train, cols)
        for feat, score in imp.items():
            rows.append({"type": typ, "feature": feat, "importance_pct": score})
    return pd.DataFrame(rows)


def importance_pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot(index="feature", columns="type", values="importance_pct").fillna(0)
