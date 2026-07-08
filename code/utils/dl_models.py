"""Deep learning forecasters using official implementations.

- Autoformer, iTransformer: thuml/Time-Series-Library (vendor submodule)
- N-HiTS, LSTM: Nixtla/neuralforecast
"""
from __future__ import annotations

import numpy as np

from .neuralforecast_adapter import train_nf_forecaster
from .tslib_adapter import train_tslib_forecaster

TSLIB_MODELS = {"Autoformer", "iTransformer"}
NF_MODELS = {"N-HiTS", "LSTM"}


def train_dl_forecaster(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    model_name: str,
    epochs: int = 100,
) -> np.ndarray:
    """Train one series forecaster and return horizon-step predictions."""
    if model_name in TSLIB_MODELS:
        return train_tslib_forecaster(series, lookback, horizon, model_name, epochs=epochs)
    if model_name in NF_MODELS:
        return train_nf_forecaster(series, lookback, horizon, model_name, epochs=epochs)
    raise ValueError(f"Unknown DL model: {model_name}")


# Backward-compatible registry (model_name string keys used by phase_experiments)
DL_MODELS = {
    "LSTM": lambda lb, h, s: train_dl_forecaster(s, lb, h, "LSTM"),
    "N-HiTS": lambda lb, h, s: train_dl_forecaster(s, lb, h, "N-HiTS"),
    "Autoformer": lambda lb, h, s: train_dl_forecaster(s, lb, h, "Autoformer"),
    "iTransformer": lambda lb, h, s: train_dl_forecaster(s, lb, h, "iTransformer"),
}
