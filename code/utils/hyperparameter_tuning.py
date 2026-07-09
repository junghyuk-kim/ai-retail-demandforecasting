"""Optuna hyperparameter tuning for XGBoost panel forecasting by store type."""
from __future__ import annotations

from typing import Any

import pandas as pd

import numpy as np

from .feature_importance import feature_cols
from .forecasting import build_scaled_panel_training, scaled_recursive_forecast
from .metrics import mape
from .splits import TRAIN_WEEK_MAX, VAL_WEEKS

# 08·13장과 동일한 기본 하이퍼파라미터 (튜닝 전 baseline)
DEFAULT_XGB_PARAMS: dict[str, Any] = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)


def build_xgb(params: dict[str, Any] | None = None):
    from xgboost import XGBRegressor

    p = {**DEFAULT_XGB_PARAMS, **(params or {})}
    return XGBRegressor(**p)  # CPU: 소규모 패널 + 재귀 단건 예측에 유리


def panel_val_mape(
    feat_df: pd.DataFrame,
    typ: str,
    params: dict[str, Any] | None = None,
) -> float:
    """type 패널(family별 Min-Max 정규화) 학습 후 검증 구간 재귀예측 평균 MAPE (%).

    엔진과 동일하게 누수 없는 재귀 다단계 예측으로 val을 평가한다.
    """
    cols = feature_cols(feat_df)
    sub = feat_df[feat_df["type"] == typ]
    keys = list(sub[["type", "family"]].drop_duplicates().itertuples(index=False, name=None))
    train_scaled, scale = build_scaled_panel_training(sub, keys, cols, TRAIN_WEEK_MAX)
    if train_scaled.empty:
        return float("nan")
    model = build_xgb(params)
    model.fit(train_scaled[cols], train_scaled["y"].astype(float))
    preds = scaled_recursive_forecast(model, sub, keys, cols, scale, TRAIN_WEEK_MAX, VAL_WEEKS, len(VAL_WEEKS))
    errs = []
    for (t, f), pred in preds.items():
        act = sub[(sub["family"] == f) & (sub["yearweek"].isin(VAL_WEEKS))].sort_values("yearweek")["sales"].to_numpy(float)
        if len(act) == len(pred) and len(pred):
            errs.append(mape(act, pred))
    return float(np.nanmean(errs)) if errs else float("nan")


def _suggest_params(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def tune_xgb_type(
    feat_df: pd.DataFrame,
    typ: str,
    n_trials: int = 30,
    seed: int = 42,
) -> tuple[dict[str, Any], float, object]:
    """단일 type Optuna study — validation MAPE 최소화."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _suggest_params(trial)
        return panel_val_mape(feat_df, typ, params)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value), study


def tune_xgb_all_types(
    feat_df: pd.DataFrame,
    types: list | None = None,
    n_trials: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """5 type 각각 Optuna 튜닝 + baseline 대비 개선율."""
    types = types or sorted(feat_df["type"].dropna().unique())
    rows = []
    for typ in types:
        baseline = panel_val_mape(feat_df, typ, None)
        best_params, best_mape, _ = tune_xgb_type(feat_df, typ, n_trials=n_trials, seed=seed)
        improve = (
            (baseline - best_mape) / baseline * 100 if baseline and baseline == baseline else float("nan")
        )
        row = {
            "type": typ,
            "baseline_mape": round(baseline, 4),
            "tuned_mape": round(best_mape, 4),
            "improvement_pct": round(improve, 2),
            "n_trials": n_trials,
            **{f"best_{k}": v for k, v in best_params.items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)
