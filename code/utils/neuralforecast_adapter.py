"""Adapter for Nixtla/neuralforecast official NHITS & LSTM."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def _series_to_panel(series: np.ndarray, unique_id: str = "s0") -> pd.DataFrame:
    y = np.asarray(series, dtype=float)
    ds = pd.date_range("2013-01-01", periods=len(y), freq="W-MON")
    return pd.DataFrame({"unique_id": unique_id, "ds": ds, "y": y})


def train_nf_forecaster(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    model_name: str,
    epochs: int = 40,
) -> np.ndarray:
    """Fit official neuralforecast NHITS or LSTM on one series."""
    y = np.asarray(series, dtype=float)
    if len(y) < lookback + horizon + 5:
        return np.full(horizon, max(float(y[-1]) if len(y) else 0.0, 0.0))

    from neuralforecast import NeuralForecast

    if model_name == "N-HiTS":
        from neuralforecast.models import NHITS

        model = NHITS(
            h=horizon,
            input_size=lookback,
            max_steps=epochs,
            scaler_type="robust",
            early_stop_patience_steps=10,
            val_check_steps=50,
            random_seed=42,
        )
        col = "NHITS"
    elif model_name == "LSTM":
        from neuralforecast.models import LSTM

        model = LSTM(
            h=horizon,
            input_size=lookback,
            max_steps=epochs,
            scaler_type="robust",
            early_stop_patience_steps=10,
            val_check_steps=50,
            random_seed=42,
        )
        col = "LSTM"
    else:
        raise ValueError(f"Unsupported neuralforecast model: {model_name}")

    df = _series_to_panel(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf = NeuralForecast(models=[model], freq="W-MON")
        nf.fit(df=df, val_size=horizon)

    fcst = nf.predict()
    pred = fcst[col].to_numpy(dtype=float)
    if len(pred) < horizon:
        pred = np.resize(pred, horizon)
    else:
        pred = pred[-horizon:]
    return np.maximum(pred, 0.0)
