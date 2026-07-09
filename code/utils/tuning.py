"""조건(type×cluster) 단위 Optuna(TPE) 하이퍼파라미터 튜닝.

논문 §4.4: TPE 샘플러로 validation 구간 RMSE를 최소화. 축소 데이터(2-type)에서는
클러스터 대표평균 시계열 대신 조건 패널 전체로 직접 튜닝(규모가 작아 가능).
튜닝은 train으로 학습→val 예측, 최종 test 예측은 train+val 재학습으로 수행한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import recursive_panel_forecast
from .metrics import rmse


def _default_xgb(params: dict | None = None):
    """XGBRegressor — Optuna 튜닝 결과(params) 반영, 없으면 기본값. GPU 자동."""
    from xgboost import XGBRegressor

    base = dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    if params:
        base.update(params)
    # 소규모 패널 + 재귀 단건 예측 → CPU가 빠르고 device 불일치 경고 없음
    return XGBRegressor(**base)


def _rmse_over_families(pred_map: dict, actual_map: dict) -> float:
    errs = []
    for k, pred in pred_map.items():
        act = actual_map.get(k)
        if act is not None and len(act) == len(pred) and len(pred):
            errs.append(rmse(act, pred))
    return float(np.mean(errs)) if errs else float("inf")


def tune_xgb_condition(
    train_df: pd.DataFrame,
    hist_frames: dict,
    future_val_frames: dict,
    val_actual: dict,
    feature_cols: list[str],
    horizon: int,
    n_trials: int = 25,
    seed: int = 42,
) -> dict:
    """XGBoost 조건 패널 Optuna 튜닝 — val 재귀예측 RMSE 최소화."""
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = train_df[feature_cols].fillna(0)
    y = train_df["sales"].astype(float)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        model = XGBRegressor(**params)
        model.fit(X, y)
        preds = recursive_panel_forecast(model, feature_cols, hist_frames, future_val_frames, horizon)
        return _rmse_over_families(preds, val_actual)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update(objective_rmse=float(study.best_value))
    return best


def tune_itransformer_condition(
    wide_train: pd.DataFrame,
    val_actual: dict,
    horizon: int,
    lookback: int,
    n_trials: int = 8,
    epochs: int = 40,
    seed: int = 42,
    model_name: str = "iTransformer",
) -> dict:
    """iTransformer/Autoformer 조건 패널 Optuna 튜닝 — val 예측 RMSE 최소화 (경량)."""
    import optuna

    from .tslib_adapter import train_tslib_condition

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    families = list(wide_train.columns)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        e_layers = trial.suggest_int("e_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        preds = train_tslib_condition(
            wide_train, horizon, lookback, model_name, epochs=epochs,
            lr=lr, d_model=d_model, e_layers=e_layers, dropout=dropout,
        )
        pred_map = {f: preds.get(f) for f in families if preds.get(f) is not None}
        return _rmse_over_families(pred_map, val_actual)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update(objective_rmse=float(study.best_value))
    return best
