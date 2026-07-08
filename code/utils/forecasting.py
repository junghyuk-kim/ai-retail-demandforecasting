"""Classical and ML forecasting wrappers."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .intermittent import forecast_sba, forecast_tsb


def forecast_naive_last(series: pd.Series, horizon: int) -> np.ndarray:
    last = float(series.iloc[-1]) if len(series) else 0.0
    return np.full(horizon, max(last, 0.0))


def forecast_arima(series: pd.Series, horizon: int) -> np.ndarray:
    y = series.astype(float).values
    if len(y) < 8:
        return forecast_naive_last(series, horizon)
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            pred = fit.forecast(horizon)
        return np.maximum(np.asarray(pred, dtype=float), 0.0)
    except Exception:
        return forecast_naive_last(series, horizon)


def forecast_prophet(weeks: pd.Series, sales: pd.Series, horizon: int) -> np.ndarray:
    try:
        from prophet import Prophet

        df = pd.DataFrame({"ds": pd.to_datetime(weeks.astype(str) + "-1", format="%G%V-%u", errors="coerce"), "y": sales.astype(float)})
        df = df.dropna()
        if len(df) < 8:
            return forecast_naive_last(sales, horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon, freq="W")
            fc = m.predict(future)
        return np.maximum(fc["yhat"].iloc[-horizon:].values.astype(float), 0.0)
    except Exception:
        return forecast_naive_last(sales, horizon)


def forecast_ml_panel(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "xgboost",
) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["sales"].astype(float)
    X_val = val_df[feature_cols].fillna(0)

    if model_name == "rf":
        model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    else:
        xgb_params = dict(
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
                xgb_params["tree_method"] = "hist"
                xgb_params["device"] = "cuda"
        except Exception:
            pass
        model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return np.maximum(pred.astype(float), 0.0)


FORECASTERS = {
    "naive": lambda s, h, **_: forecast_naive_last(s, h),
    "arima": lambda s, h, **_: forecast_arima(s, h),
    "sba": lambda s, h, **_: forecast_sba(s, h),
    "tsb": lambda s, h, **_: forecast_tsb(s, h),
}
