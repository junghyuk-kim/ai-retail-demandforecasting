"""Adapter for Nixtla/neuralforecast official NHITS & LSTM."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def _yearweek_to_date(yw: int) -> pd.Timestamp:
    year, week = int(yw) // 100, int(yw) % 100
    return pd.Timestamp.fromisocalendar(year, week, 1)


def _nf_model(model_name: str, horizon: int, lookback: int, epochs: int):
    from neuralforecast.models import LSTM, NHITS

    common = dict(
        h=horizon,
        input_size=lookback,
        max_steps=epochs,
        scaler_type="robust",
        early_stop_patience_steps=min(10, max(3, epochs // 4)),
        val_check_steps=max(10, epochs),
        random_seed=42,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    if model_name == "N-HiTS":
        return NHITS(**common), "NHITS"
    if model_name == "LSTM":
        return LSTM(**common), "LSTM"
    raise ValueError(f"Unsupported neuralforecast model: {model_name}")


def train_nf_condition(
    wide_train: pd.DataFrame,
    horizon: int,
    lookback: int,
    model_name: str,
    epochs: int = 40,
) -> dict[str, np.ndarray]:
    """조건 내 여러 family를 한 번에 학습 (공식 NHITS/LSTM)."""
    if wide_train.shape[0] < lookback + horizon + 5 or wide_train.shape[1] == 0:
        return {}

    from neuralforecast import NeuralForecast

    rows = []
    for fam in wide_train.columns:
        for yw, val in wide_train[fam].items():
            rows.append(
                {
                    "unique_id": str(fam),
                    "ds": _yearweek_to_date(int(yw)),
                    "y": float(val),
                }
            )
    df = pd.DataFrame(rows)
    model, col = _nf_model(model_name, horizon, lookback, epochs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf = NeuralForecast(models=[model], freq="W-MON")
        nf.fit(df=df, val_size=horizon)

    fcst = nf.predict()
    out: dict[str, np.ndarray] = {}
    for fam in wide_train.columns:
        uid = str(fam)
        sub = fcst[fcst["unique_id"] == uid][col]
        pred = sub.tail(horizon).to_numpy(dtype=float)
        if len(pred) != horizon:
            pred = np.resize(pred, horizon)
        out[fam] = np.maximum(pred, 0.0)
    return out


def _nf_model_static(model_name: str, horizon: int, lookback: int, epochs: int, stat_cols: list[str]):
    """정적 외생변수(임베딩) 지원 NHITS/LSTM (Phase2 하이브리드용)."""
    from neuralforecast.models import LSTM, NHITS

    common = dict(
        h=horizon,
        input_size=lookback,
        max_steps=epochs,
        scaler_type="robust",
        stat_exog_list=stat_cols,
        early_stop_patience_steps=min(10, max(3, epochs // 4)),
        val_check_steps=max(10, epochs),
        random_seed=42,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    if model_name == "N-HiTS":
        return NHITS(**common), "NHITS"
    if model_name == "LSTM":
        return LSTM(**common), "LSTM"
    raise ValueError(f"Unsupported neuralforecast model: {model_name}")


def train_nf_condition_static(
    wide_train: pd.DataFrame,
    embedding_map: dict,
    horizon: int,
    lookback: int,
    model_name: str,
    epochs: int = 40,
) -> dict[str, np.ndarray]:
    """조건 내 다변량 패널 + family별 정적 임베딩(static exog) 결합 학습 (Phase2 하이브리드).

    embedding_map: {family: np.ndarray[dim]} — 06/10장 시계열 임베딩.
    임베딩을 unique_id(family)별 static 특성으로 넣어 LSTM/NHITS가 결합 학습한다.
    """
    if wide_train.shape[0] < lookback + horizon + 5 or wide_train.shape[1] == 0:
        return {}
    from neuralforecast import NeuralForecast

    dim = next(iter(embedding_map.values())).shape[0]
    stat_cols = [f"emb_{i}" for i in range(dim)]

    rows, static_rows = [], []
    for fam in wide_train.columns:
        emb = embedding_map.get(fam)
        if emb is None:
            continue
        for yw, val in wide_train[fam].items():
            rows.append({"unique_id": str(fam), "ds": _yearweek_to_date(int(yw)), "y": float(val)})
        static_rows.append({"unique_id": str(fam), **{f"emb_{i}": float(emb[i]) for i in range(dim)}})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    static_df = pd.DataFrame(static_rows)

    model, col = _nf_model_static(model_name, horizon, lookback, epochs, stat_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf = NeuralForecast(models=[model], freq="W-MON")
        nf.fit(df=df, static_df=static_df, val_size=horizon)
    fcst = nf.predict()

    out: dict[str, np.ndarray] = {}
    for fam in wide_train.columns:
        sub = fcst[fcst["unique_id"] == str(fam)][col]
        pred = sub.tail(horizon).to_numpy(dtype=float)
        if len(pred) != horizon:
            pred = np.resize(pred, horizon) if len(pred) else np.zeros(horizon)
        out[fam] = np.maximum(pred, 0.0)
    return out


def train_nf_forecaster(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    model_name: str,
    epochs: int = 40,
) -> np.ndarray:
    """단일 시계열 fallback."""
    y = np.asarray(series, dtype=float)
    if len(y) < lookback + horizon + 5:
        return np.full(horizon, max(float(y[-1]) if len(y) else 0.0, 0.0))

    wide = pd.DataFrame({"s0": y})
    preds = train_nf_condition(wide, horizon, lookback, model_name, epochs=epochs)
    if not preds:
        return np.full(horizon, max(float(y[-1]), 0.0))
    return preds["s0"]
