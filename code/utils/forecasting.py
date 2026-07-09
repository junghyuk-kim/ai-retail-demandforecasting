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


def _recompute_lag_features(sales_hist: list[float], feat_row: dict) -> dict:
    """진행 중인 판매 이력으로 lag/rolling/간헐성 피처를 재계산 (03장 정의와 동일).

    누수 방지: test 구간 예측 시 lag_k·rolling은 실제 미래값이 아니라
    직전까지의 (실측+예측) 이력에서 계산한다.
    """
    s = np.asarray(sales_hist, dtype=float)
    n = len(s)

    def lag(k):
        return s[n - k] if n >= k else np.nan

    def roll_mean(w):
        return s[max(0, n - w):].mean() if n else np.nan

    def roll_std(w):
        window = s[max(0, n - w):]
        return window.std(ddof=1) if len(window) > 1 else np.nan

    out = dict(feat_row)  # 외생·달력 피처는 그대로 (미래에도 알려진 값)
    for k in (1, 2, 4, 8):
        out[f"lag_{k}"] = lag(k)
    for w in (4, 8, 12):
        out[f"roll_mean_{w}"] = roll_mean(w)
        out[f"roll_std_{w}"] = roll_std(w)
    if n:
        window12 = s[max(0, n - 12):]
        out["zero_ratio_12"] = float((window12 == 0).mean())
    return out


def _minmax_params(sales: list[float]) -> tuple[float, float]:
    lo = min(sales)
    rng = (max(sales) - lo) or 1.0
    return lo, rng


def build_scaled_panel_training(
    feat_df: pd.DataFrame, keys: list[tuple], feature_cols: list[str], hist_max: int
) -> tuple[pd.DataFrame, dict]:
    """family별 Min-Max(판매량)로 정규화한 패널 학습 데이터 생성 (논문 §4.2 forecasting scaling).

    이질적 규모(예: 판매 1 vs 265,000) family를 한 패널에 학습할 때 작은 family를
    과대예측하는 문제를 방지. lag/rolling은 정규화 판매 이력에서 재계산, 타깃도 정규화.
    반환: (학습 DataFrame[feature_cols + 'y'], {(t,f): (lo, rng)})
    """
    rows, scale = [], {}
    for t, f in keys:
        g = feat_df[(feat_df["type"] == t) & (feat_df["family"] == f) & (feat_df["yearweek"] <= hist_max)].sort_values("yearweek")
        s = g["sales"].astype(float).tolist()
        if len(s) < 2:
            scale[(t, f)] = (0.0, 1.0)
            continue
        lo, rng = _minmax_params(s)
        scale[(t, f)] = (lo, rng)
        recs = g.to_dict("records")
        sc_s = [(v - lo) / rng for v in s]
        for i in range(1, len(s)):
            ft = _recompute_lag_features(sc_s[:i], recs[i])
            rows.append({**{c: ft.get(c, 0.0) for c in feature_cols}, "y": sc_s[i]})
    return pd.DataFrame(rows).fillna(0), scale


def scaled_recursive_forecast(
    model, feat_df: pd.DataFrame, keys: list[tuple], feature_cols: list[str],
    scale: dict, hist_max: int, future_weeks: list[int], horizon: int,
) -> dict[tuple, np.ndarray]:
    """정규화 공간에서 재귀 다단계 예측 후 family별 역 Min-Max (원 스케일 반환)."""
    preds = {}
    for t, f in keys:
        lo, rng = scale.get((t, f), (0.0, 1.0))
        g = feat_df[(feat_df["type"] == t) & (feat_df["family"] == f)]
        hist = g[g["yearweek"] <= hist_max].sort_values("yearweek")["sales"].astype(float).tolist()
        fut = g[g["yearweek"].isin(future_weeks)].sort_values("yearweek").head(horizon)
        if not hist or fut.empty:
            continue
        sh = [(v - lo) / rng for v in hist]
        out = []
        for _, fr in fut.iterrows():
            ft = _recompute_lag_features(sh, fr.to_dict())
            x = pd.DataFrame([{c: ft.get(c, 0.0) for c in feature_cols}]).fillna(0)
            p = float(model.predict(x)[0])
            sh.append(p)
            out.append(max(p * rng + lo, 0.0))
        preds[(t, f)] = np.array(out, dtype=float)
    return preds


def recursive_panel_forecast(
    model,
    feature_cols: list[str],
    hist_frames: dict[tuple, pd.DataFrame],
    future_frames: dict[tuple, pd.DataFrame],
    horizon: int,
) -> dict[tuple, np.ndarray]:
    """학습된 패널 모델로 조건 내 각 family를 재귀 다단계 예측.

    hist_frames[(t,f)]  : 학습 상한까지의 실측 (yearweek, sales, 외생피처)
    future_frames[(t,f)]: 예측 구간의 외생·달력 피처 (sales 제외), yearweek 오름차순
    반환: {(t,f): pred[horizon]}
    """
    preds: dict[tuple, np.ndarray] = {}
    for key, fut in future_frames.items():
        hist = hist_frames.get(key)
        if hist is None or fut.empty:
            continue
        sales_hist = hist.sort_values("yearweek")["sales"].astype(float).tolist()
        fut = fut.sort_values("yearweek")
        yhat = []
        for _, frow in fut.head(horizon).iterrows():
            feat = _recompute_lag_features(sales_hist, frow.to_dict())
            x = pd.DataFrame([{c: feat.get(c, 0.0) for c in feature_cols}]).fillna(0)
            p = float(np.maximum(model.predict(x)[0], 0.0))
            yhat.append(p)
            sales_hist.append(p)  # 다음 스텝 lag 갱신용
        preds[key] = np.array(yhat, dtype=float)
    return preds


FORECASTERS = {
    "naive": lambda s, h, **_: forecast_naive_last(s, h),
    "arima": lambda s, h, **_: forecast_arima(s, h),
    "sba": lambda s, h, **_: forecast_sba(s, h),
    "tsb": lambda s, h, **_: forecast_tsb(s, h),
}
