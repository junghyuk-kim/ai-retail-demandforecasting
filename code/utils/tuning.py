"""조건(type×cluster) 단위 Optuna(TPE) 하이퍼파라미터 튜닝.

논문 §4.4: TPE 샘플러로 validation 구간 RMSE를 최소화. 축소 데이터(2-type)에서는
클러스터 대표평균 시계열 대신 조건 패널 전체로 직접 튜닝(규모가 작아 가능).
튜닝은 train으로 학습→val 예측, 최종 test 예측은 train+val 재학습으로 수행한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecasting import build_scaled_panel_training, scaled_recursive_forecast
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
    feat_df: pd.DataFrame,
    keys: list,
    feature_cols: list[str],
    train_max: int,
    val_weeks: list,
    val_actual: dict,
    horizon: int,
    n_trials: int = 25,
    seed: int = 42,
) -> dict:
    """XGBoost 조건 패널 Optuna 튜닝 — family별 Min-Max 정규화 후 val 재귀예측 RMSE 최소화."""
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_df, scale = build_scaled_panel_training(feat_df, keys, feature_cols, train_max)
    if train_df.empty:
        return {}
    X = train_df[feature_cols]
    y = train_df["y"].astype(float)

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
        model = XGBRegressor(**params).fit(X, y)
        preds = scaled_recursive_forecast(
            model, feat_df, keys, feature_cols, scale, train_max, val_weeks, horizon
        )
        return _rmse_over_families(preds, val_actual)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update(objective_rmse=float(study.best_value))
    return best


DEFAULT_LSTM_PARAMS = dict(
    input_size=26, encoder_hidden_size=128, encoder_n_layers=2,
    learning_rate=1e-3, max_steps=300,
)


def lstm_embedding_val_mape(
    wide_train: pd.DataFrame, embedding_map: dict, val_actual: dict,
    horizon: int, params: dict | None = None,
) -> float:
    """LSTM + 임베딩(static exog)을 type/조건 패널에 학습 → val 구간 평균 family MAPE."""
    import numpy as np

    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM

    from .metrics import mape
    from .neuralforecast_adapter import build_nf_frames

    df, static_df, stat_cols = build_nf_frames(wide_train, embedding_map)
    if df.empty:
        return float("nan")
    p = {**DEFAULT_LSTM_PARAMS, **(params or {})}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LSTM(
            h=horizon, stat_exog_list=stat_cols, scaler_type="robust",
            random_seed=42, accelerator="gpu", devices=1,
            enable_progress_bar=False, logger=False,
            early_stop_patience_steps=-1, **p,
        )
        nf = NeuralForecast(models=[model], freq="W-MON")
        nf.fit(df=df, static_df=static_df, val_size=horizon)
        fcst = nf.predict()
    errs = []
    for fam in wide_train.columns:
        act = val_actual.get(fam)
        pred = fcst[fcst["unique_id"] == str(fam)]["LSTM"].tail(horizon).to_numpy(dtype=float)
        if act is not None and len(pred) == len(act) and len(act):
            errs.append(mape(act, np.maximum(pred, 0.0)))
    return float(np.nanmean(errs)) if errs else float("nan")


def tune_lstm_embedding_type(
    wide_train: pd.DataFrame, embedding_map: dict, val_actual: dict,
    horizon: int, n_trials: int = 12, seed: int = 42,
) -> tuple[dict, float]:
    """LSTM+임베딩(대표모델) Optuna 튜닝 — val MAPE 최소화. (best_params, best_mape)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            input_size=trial.suggest_categorical("input_size", [13, 26, 39]),
            encoder_hidden_size=trial.suggest_categorical("encoder_hidden_size", [64, 128, 256]),
            encoder_n_layers=trial.suggest_int("encoder_n_layers", 1, 3),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            max_steps=trial.suggest_categorical("max_steps", [200, 400, 600]),
        )
        return lstm_embedding_val_mape(wide_train, embedding_map, val_actual, horizon, params)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


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
