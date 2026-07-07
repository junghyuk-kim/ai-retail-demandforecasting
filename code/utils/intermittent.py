"""Intermittent demand forecasting: Croston family SBA, TSB."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_sba(series: pd.Series, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Syntetos-Boylan Approximation."""
    y = series.astype(float).values
    if len(y) == 0:
        return np.zeros(horizon)
    z, p = y[0] if y[0] > 0 else 0.0, 1.0
    q = 1.0
    for i in range(1, len(y)):
        if y[i] > 0:
            z = alpha * y[i] + (1 - alpha) * z
            q = alpha * p + (1 - alpha) * q
            p = 1.0
        else:
            p += 1.0
    bias = 1.0 - alpha / (2.0 - alpha)
    forecast_level = bias * z / max(q, 1e-8)
    return np.full(horizon, max(forecast_level, 0.0))


def forecast_tsb(series: pd.Series, horizon: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """Teunter-Syntetos-Babai method."""
    y = series.astype(float).values
    if len(y) == 0:
        return np.zeros(horizon)
    prob, z = 0.0, 0.0
    for val in y:
        if val > 0:
            prob = beta * 1.0 + (1 - beta) * prob
            z = alpha * val + (1 - alpha) * z
        else:
            prob = (1 - beta) * prob
    level = prob * z
    return np.full(horizon, max(level, 0.0))
