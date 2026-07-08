"""Deep learning forecasters — official model code via adapters.

- Autoformer, iTransformer: thuml/Time-Series-Library (vendor submodule)
- N-HiTS, LSTM: Nixtla/neuralforecast (pip)

Training is per type×cluster multivariate panel (not per-SKU loops).
See docs/실험_프레임워크.md for differences vs thesis experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .neuralforecast_adapter import train_nf_condition, train_nf_forecaster
from .tslib_adapter import train_tslib_condition, train_tslib_forecaster

TSLIB_MODELS = {"Autoformer", "iTransformer"}
NF_MODELS = {"N-HiTS", "LSTM"}
DL_MODEL_NAMES = TSLIB_MODELS | NF_MODELS


def condition_wide(df: pd.DataFrame) -> pd.DataFrame:
    """type×cluster 조건 내 family×week 판매 wide matrix."""
    wide = df.pivot_table(index="yearweek", columns="family", values="sales", aggfunc="sum")
    return wide.sort_index().fillna(0)


def forecast_dl_condition(
    wide_train: pd.DataFrame,
    horizon: int,
    lookback: int,
    models: list[str],
    epochs: int = 40,
) -> dict[str, dict[str, np.ndarray]]:
    """조건 단위 DL 학습 — model → {family: pred[horizon]}."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for model_name in models:
        if model_name in TSLIB_MODELS:
            out[model_name] = train_tslib_condition(
                wide_train, horizon, lookback, model_name, epochs=epochs
            )
        elif model_name in NF_MODELS:
            out[model_name] = train_nf_condition(
                wide_train, horizon, lookback, model_name, epochs=epochs
            )
        else:
            raise ValueError(f"Unknown DL model: {model_name}")
    return out


def train_dl_forecaster(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    model_name: str,
    epochs: int = 100,
) -> np.ndarray:
    """단일 시계열 fallback."""
    if model_name in TSLIB_MODELS:
        return train_tslib_forecaster(series, lookback, horizon, model_name, epochs=epochs)
    if model_name in NF_MODELS:
        return train_nf_forecaster(series, lookback, horizon, model_name, epochs=epochs)
    raise ValueError(f"Unknown DL model: {model_name}")


DL_MODELS = {
    "LSTM": lambda lb, h, s: train_dl_forecaster(s, lb, h, "LSTM"),
    "N-HiTS": lambda lb, h, s: train_dl_forecaster(s, lb, h, "N-HiTS"),
    "Autoformer": lambda lb, h, s: train_dl_forecaster(s, lb, h, "Autoformer"),
    "iTransformer": lambda lb, h, s: train_dl_forecaster(s, lb, h, "iTransformer"),
}
