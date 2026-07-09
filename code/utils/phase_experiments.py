"""Phase 1 / Phase 2 forecasting experiments (40 conditions x 10 algos)."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .dl_models import DL_MODEL_NAMES, NF_MODELS, TSLIB_MODELS, condition_wide
from .embeddings import EMBEDDERS
from .forecasting import (
    FORECASTERS,
    build_scaled_panel_training,
    forecast_arima,
    forecast_ml_panel,
    forecast_prophet,
    recursive_panel_forecast,
    scaled_recursive_forecast,
)
from .intermittent import forecast_sba, forecast_tsb
from .metrics import FORECAST_METRICS, forecast_metrics, wmape
from .neuralforecast_adapter import train_nf_condition, train_nf_condition_static
from .splits import TEST_WEEKS, TRAIN_WEEK_MAX, TRAINVAL_WEEK_MAX, VAL_WEEKS
from .tslib_adapter import train_tslib_condition
from .tuning import tune_itransformer_condition, tune_xgb_condition

STAT_MODELS = ["ARIMA", "Prophet", "SBA", "TSB"]
ML_MODELS = ["RF", "XGBoost"]
DL_MODELS = ["LSTM", "Autoformer", "N-HiTS", "iTransformer"]

PHASE1_MODELS = STAT_MODELS + ML_MODELS + DL_MODELS
RANK_METRIC = "mape"  # 조건별 Best 선정 기준 (MAE/RMSE/MAPE/MASE 중)

EMBEDDING_NAMES = list(EMBEDDERS.keys())
# Phase2 임베딩 하이브리드 대표 base. LSTM은 Phase1 상위(mean MAPE 최저)이면서
# 임베딩을 static exog로 결합할 수 있는 신경망 → 논문(iTransformer+임베딩)에 더 근접.
REPRESENTATIVE_BASE_MODEL = "LSTM"
LOOKBACK = 26  # 13주 예측 지평에 맞춘 lookback (약 반년)
try:
    import torch

    DL_EPOCHS = 60 if torch.cuda.is_available() else 40
except Exception:
    DL_EPOCHS = 40
N_EMB_DIM = 10

# 튜닝 설정 (논문 §4.4 Optuna TPE, val RMSE). 축소 데이터라 조건 패널 직접 튜닝.
TUNE_XGB = True
TUNE_XGB_TRIALS = 25
TUNE_DL = True
TUNE_DL_TRIALS = 8
TUNE_DL_EPOCHS = 40  # 튜닝 시 epoch (최종은 DL_EPOCHS)

SERIES_MODELS = {"ARIMA", "Prophet", "SBA", "TSB"}
PANEL_MODELS = {"RF", "XGBoost"}


def _horizon() -> int:
    return len(TEST_WEEKS)


def build_conditions(types: list, clusters: list[int] = (1, 2, 3, 4)) -> pd.DataFrame:
    rows = [
        {"type": t, "cluster": c}
        for t in types
        for c in clusters
    ]
    return pd.DataFrame(rows)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"sales", "type", "family", "year", "week", "yearweek", "split"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _attach_embeddings(df: pd.DataFrame, emb_map: dict[tuple, np.ndarray], prefix: str = "emb") -> pd.DataFrame:
    out = df.copy()
    dim = next(iter(emb_map.values())).shape[0]
    for i in range(dim):
        out[f"{prefix}_{i}"] = out.apply(lambda r: emb_map[(r["type"], r["family"])][i], axis=1)
    return out


def _fit_embedding(train_matrix: np.ndarray, embed_fn: Callable, n_components: int = N_EMB_DIM) -> np.ndarray:
    from .embeddings import embed_pca

    n = train_matrix.shape[0]
    if n < 3:
        d = min(n_components, max(n - 1, 1))
        return embed_pca(train_matrix, n_components=d)
    d = min(n_components, n - 1, train_matrix.shape[1])
    d = max(d, 2)
    if d % 2 != 0:
        d = max(d - 1, 2)
    d = min(d, n - 1)
    try:
        return embed_fn(train_matrix, n_components=d)
    except Exception:
        return embed_pca(train_matrix, n_components=min(d, n - 1))


def _forecast_stat(
    model: str,
    weeks: pd.Series,
    sales: pd.Series,
    horizon: int,
    exog: np.ndarray | None = None,
) -> np.ndarray:
    y = sales.astype(float).values
    if model == "Prophet":
        return forecast_prophet(weeks, sales, horizon)
    if model == "ARIMA" and exog is not None and len(y) >= 8:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            ex = np.asarray(exog, dtype=float)
            if ex.ndim == 1:
                ex = ex.reshape(-1, 1)
            ex_f = np.tile(ex[-1:], (horizon, 1)) if len(ex) else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    y,
                    exog=ex,
                    order=(1, 0, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                pred = fit.forecast(horizon, exog=ex_f)
            return np.maximum(np.asarray(pred, dtype=float), 0.0)
        except Exception:
            pass
    if model == "ARIMA":
        return forecast_arima(sales, horizon)
    if model == "SBA":
        return forecast_sba(sales, horizon)
    if model == "TSB":
        return forecast_tsb(sales, horizon)
    return forecast_arima(sales, horizon)


def _hybrid_residual_adjust(
    train_sales: pd.Series,
    train_weeks: pd.Series,
    base_model: str,
    embedding: np.ndarray,
    horizon: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Train embedding -> residual correction on in-sample base forecast."""
    from sklearn.linear_model import Ridge

    y = train_sales.astype(float).values
    if len(y) < horizon + 5:
        return lambda pred: pred
    preds, actuals, embs = [], [], []
    for i in range(horizon, len(y)):
        sub_w = train_weeks.iloc[:i]
        sub_s = train_sales.iloc[:i]
        p = _forecast_stat(base_model, sub_w, sub_s, horizon=1)[0]
        preds.append(p)
        actuals.append(y[i])
        embs.append(embedding)
    if len(preds) < 5:
        return lambda pred: pred
    resid = np.array(actuals) - np.array(preds)
    reg = Ridge(alpha=1.0).fit(np.array(embs), resid)
    return lambda pred: pred + reg.predict(embedding.reshape(1, -1))[0]


def forecast_one_series(
    model: str,
    g_train: pd.DataFrame,
    g_val: pd.DataFrame,
    feature_cols: list[str],
    lookback: int = LOOKBACK,
    embedding: np.ndarray | None = None,
    hybrid: bool = False,
) -> np.ndarray:
    horizon = len(g_val)
    weeks = g_train["yearweek"]
    sales = g_train["sales"]

    if model in PANEL_MODELS:
        return np.array([])  # panel handled separately

    if model in DL_MODEL_NAMES:
        raise RuntimeError("DL models must be trained at condition level via forecast_dl_condition()")

    exog = embedding if hybrid and embedding is not None else None
    pred = _forecast_stat(model, weeks, sales, horizon, exog=exog)

    if hybrid and embedding is not None and model in {"SBA", "TSB", "Prophet"}:
        adj = _hybrid_residual_adjust(sales, weeks, model, embedding, horizon)
        pred = adj(pred)

    if len(pred) != horizon:
        pred = np.resize(pred, horizon)
    return pred


def _metrics_row(base: dict, y_true, y_pred, y_train) -> dict:
    row = dict(base)
    row.update(forecast_metrics(y_true, y_pred, y_train))
    row["wmape"] = wmape(y_true, y_pred)  # 11장 SBC vs ML 비교용 (Best 선정에는 미사용)
    return row


def _actual_map(df: pd.DataFrame, keys: list[tuple], weeks: list[int]) -> dict[tuple, np.ndarray]:
    """{(type,family): 해당 주차 실측 sales 배열} (yearweek 오름차순)."""
    out = {}
    for t, f in keys:
        g = df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"].isin(weeks))]
        out[(t, f)] = g.sort_values("yearweek")["sales"].to_numpy(dtype=float)
    return out


def _hist_future_frames(
    feat_df: pd.DataFrame, keys: list[tuple], hist_max: int, future_weeks: list[int]
):
    """재귀 예측용 hist(≤hist_max 실측)·future(예측구간 외생피처) 프레임."""
    hist, future = {}, {}
    for t, f in keys:
        base = feat_df[(feat_df["type"] == t) & (feat_df["family"] == f)]
        hist[(t, f)] = base[base["yearweek"] <= hist_max].sort_values("yearweek")
        future[(t, f)] = base[base["yearweek"].isin(future_weeks)].sort_values("yearweek")
    return hist, future


def run_phase1_condition(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_scheme: str,
    cluster_col: str,
    typ: str,
    cluster: int,
    models: list[str] | None = None,
    tune: bool = True,
) -> pd.DataFrame:
    """조건(type×cluster) Phase1: train으로 학습·튜닝(val RMSE) → train+val 재학습 → test 예측."""
    models = models or PHASE1_MODELS
    feature_cols = _feature_cols(feat_df)
    horizon = _horizon()
    rows = []

    series_list = list(
        df[(df["type"] == typ) & (df[cluster_col] == cluster)][["type", "family"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if not series_list:
        return pd.DataFrame()

    panel_models = [m for m in models if m in PANEL_MODELS]
    dl_models = [m for m in models if m in DL_MODEL_NAMES]
    series_models = [m for m in models if m in SERIES_MODELS]

    test_actual = _actual_map(df, series_list, TEST_WEEKS)
    val_actual = _actual_map(df, series_list, VAL_WEEKS)
    trainval_sales = {
        (t, f): df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAINVAL_WEEK_MAX)]
        .sort_values("yearweek")["sales"].to_numpy(dtype=float)
        for t, f in series_list
    }

    # ---------- Panel ML (RF/XGBoost): family별 Min-Max 정규화 → 튜닝 → 재학습 → test 재귀예측 ----------
    panel_preds: dict[str, dict[tuple, np.ndarray]] = {m: {} for m in panel_models}
    if panel_models:
        cond_feat = feat_df[(feat_df["type"] == typ) & (feat_df[cluster_col] == cluster)]
        train_scaled, scale_tv = build_scaled_panel_training(cond_feat, series_list, feature_cols, TRAINVAL_WEEK_MAX)
        for pm in panel_models:
            if train_scaled.empty:
                continue
            params = None
            if pm == "XGBoost" and tune and TUNE_XGB:
                try:
                    tuned = tune_xgb_condition(
                        cond_feat, series_list, feature_cols, TRAIN_WEEK_MAX, VAL_WEEKS,
                        val_actual, horizon, n_trials=TUNE_XGB_TRIALS,
                    )
                    params = {k: v for k, v in tuned.items() if k != "objective_rmse"} or None
                except Exception:
                    params = None
            model = _fit_scaled_panel(train_scaled, feature_cols, pm, params)
            panel_preds[pm] = scaled_recursive_forecast(
                model, cond_feat, series_list, feature_cols, scale_tv, TRAINVAL_WEEK_MAX, TEST_WEEKS, horizon
            )

    # ---------- DL: 튜닝(tslib) → train+val 재학습 → test 예측 ----------
    dl_preds: dict[str, dict[tuple, np.ndarray]] = {m: {} for m in dl_models}
    if dl_models:
        g_train = df[(df["type"] == typ) & (df[cluster_col] == cluster) & (df["yearweek"] <= TRAIN_WEEK_MAX)]
        g_trainval = df[(df["type"] == typ) & (df[cluster_col] == cluster) & (df["yearweek"] <= TRAINVAL_WEEK_MAX)]
        wide_train = condition_wide(g_train)
        wide_trainval = condition_wide(g_trainval)
        val_actual_by_fam = {f: val_actual.get((typ, f)) for (_, f) in series_list}
        for dm in dl_models:
            preds_fam = {}
            if dm in TSLIB_MODELS:
                params = {}
                # 튜닝은 논문 주역 iTransformer에만 적용 (Autoformer는 정규화 기본 설정)
                if tune and TUNE_DL and dm == "iTransformer" and not wide_train.empty:
                    try:
                        tuned = tune_itransformer_condition(
                            wide_train, val_actual_by_fam, horizon, LOOKBACK,
                            n_trials=TUNE_DL_TRIALS, epochs=TUNE_DL_EPOCHS, model_name=dm,
                        )
                        params = {k: v for k, v in tuned.items() if k != "objective_rmse"}
                    except Exception:
                        params = {}
                if not wide_trainval.empty:
                    preds_fam = train_tslib_condition(
                        wide_trainval, horizon, LOOKBACK, dm, epochs=DL_EPOCHS, **params
                    )
            else:  # N-HiTS / LSTM (neuralforecast)
                if not wide_trainval.empty:
                    preds_fam = train_nf_condition(wide_trainval, horizon, LOOKBACK, dm, epochs=DL_EPOCHS)
            for f, pred in preds_fam.items():
                dl_preds[dm][(typ, f)] = pred

    # ---------- Series models (ARIMA/Prophet/SBA/TSB): train+val fit → test ----------
    for t, f in series_list:
        g = df[(df["type"] == t) & (df["family"] == f)]
        g_trainval = g[g["yearweek"] <= TRAINVAL_WEEK_MAX]
        g_test = g[g["yearweek"].isin(TEST_WEEKS)].sort_values("yearweek")
        if g_test.empty:
            continue
        y_true = test_actual[(t, f)]
        y_train = trainval_sales[(t, f)]
        base = {"phase": 1, "cluster_scheme": cluster_scheme, "type": t, "cluster": cluster, "family": f}

        for model in series_models:
            pred = forecast_one_series(model, g_trainval, g_test, feature_cols)
            rows.append(_metrics_row({**base, "model": model}, y_true, pred, y_train))

        for model in dl_models:
            pred = dl_preds.get(model, {}).get((t, f))
            if pred is None or len(pred) != len(y_true):
                pred = np.zeros(len(y_true))
            rows.append(_metrics_row({**base, "model": model}, y_true, pred, y_train))

        for model in panel_models:
            pred = panel_preds.get(model, {}).get((t, f))
            if pred is None or len(pred) != len(y_true):
                pred = np.zeros(len(y_true))
            rows.append(_metrics_row({**base, "model": model}, y_true, pred, y_train))

    return pd.DataFrame(rows)


def _fit_scaled_panel(train_scaled: pd.DataFrame, feature_cols: list[str], model_name: str, params: dict | None):
    """정규화 패널 학습 (타깃 'y' = 정규화 판매). params: XGBoost Optuna 결과."""
    from sklearn.ensemble import RandomForestRegressor

    X = train_scaled[feature_cols]
    y = train_scaled["y"].astype(float)
    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        from .tuning import _default_xgb

        model = _default_xgb(params)
    model.fit(X, y)
    return model


def run_phase1_all(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_col: str,
    cluster_scheme: str,
    models: list[str] | None = None,
) -> pd.DataFrame:
    models = models or PHASE1_MODELS
    types = sorted(df["type"].unique())
    conditions = build_conditions(types)
    parts = []
    for row in tqdm(conditions.itertuples(index=False), total=len(conditions), desc=f"Phase1 {cluster_scheme}"):
        part = run_phase1_condition(
            df, feat_df, cluster_scheme, cluster_col, row.type, int(row.cluster), models=models
        )
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _summarize_by_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    rank_metric: str = RANK_METRIC,
    model_col: str = "model",
):
    condition_cols = ["cluster_scheme", "type", "cluster"]
    agg = {m: "mean" for m in FORECAST_METRICS}
    summary = df.groupby(group_cols, as_index=False).agg(agg)
    summary = summary.rename(columns={m: f"{m}_mean" for m in FORECAST_METRICS})
    idx = summary.groupby(condition_cols)[f"{rank_metric}_mean"].idxmin()
    best = summary.loc[idx].copy()
    rename = {model_col: "best_model", f"{rank_metric}_mean": f"best_{rank_metric}"}
    best = best.rename(columns=rename)
    return summary, best


def summarize_phase1(phase1_df: pd.DataFrame, rank_metric: str = RANK_METRIC) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["cluster_scheme", "type", "cluster", "model"]
    return _summarize_by_metric(phase1_df, group_cols, rank_metric)


def merge_phase1_best(
    stat_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    dl_df: pd.DataFrame,
    rank_metric: str = RANK_METRIC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """07·08·09 결과를 합쳐 40조건×10모델 중 Best 선정."""
    combined = pd.concat([stat_df, ml_df, dl_df], ignore_index=True)
    summary, best = summarize_phase1(combined, rank_metric=rank_metric)
    return summary, best


def build_global_embedding_cache(df: pd.DataFrame) -> dict[str, dict[tuple, np.ndarray]]:
    """165개 전체 시계열에 대해 6종 임베딩을 1회만 계산 (Phase2 가속)."""
    keys_df = df[["type", "family"]].drop_duplicates().sort_values(["type", "family"])
    keys = list(zip(keys_df["type"], keys_df["family"]))
    train_rows = []
    for t, f in keys:
        s = df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAINVAL_WEEK_MAX)]["sales"].values
        train_rows.append(s)
    train_matrix = np.vstack(train_rows).astype(float)
    cache: dict[str, dict[tuple, np.ndarray]] = {}
    for emb_name, embed_fn in tqdm(EMBEDDERS.items(), desc="Global embeddings"):
        emb_all = _fit_embedding(train_matrix, embed_fn)
        cache[emb_name] = {keys[i]: emb_all[i] for i in range(len(keys))}
    return cache


def run_phase2_condition(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_scheme: str,
    cluster_col: str,
    typ: str,
    cluster: int,
    best_model: str,
    embedding_name: str,
    emb_cache: dict[str, dict[tuple, np.ndarray]] | None = None,
) -> pd.DataFrame:
    feature_cols = _feature_cols(feat_df)
    rows = []
    series_list = (
        df[(df["type"] == typ) & (df[cluster_col] == cluster)][["type", "family"]]
        .drop_duplicates()
        .values
        .tolist()
    )
    if not series_list:
        return pd.DataFrame()

    if emb_cache is not None:
        emb_map = emb_cache[embedding_name]
    else:
        train_rows = []
        keys = []
        for t, f in series_list:
            s = df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAINVAL_WEEK_MAX)]["sales"].values
            train_rows.append(s)
            keys.append((t, f))
        train_matrix = np.vstack(train_rows).astype(float)
        emb_all = _fit_embedding(train_matrix, EMBEDDERS[embedding_name])
        emb_map = {keys[i]: emb_all[i] for i in range(len(keys))}

    combo = f"{best_model}+{embedding_name}"
    emb_dim = next(iter(emb_map.values())).shape[0]
    keys = [tuple(k) for k in series_list]
    test_actual = _actual_map(df, keys, TEST_WEEKS)
    trainval_sales = {
        (t, f): df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAINVAL_WEEK_MAX)]
        .sort_values("yearweek")["sales"].to_numpy(dtype=float)
        for t, f in keys
    }
    base_row = lambda t, f: {
        "phase": 2, "cluster_scheme": cluster_scheme, "type": t, "cluster": cluster,
        "family": f, "model": combo, "base_model": best_model, "embedding": embedding_name,
    }

    if best_model in NF_MODELS:
        # LSTM/N-HiTS + 임베딩(static exog) 하이브리드 — 논문 iTransformer+임베딩에 근접
        g_trainval = df[(df["type"] == typ) & (df[cluster_col] == cluster) & (df["yearweek"] <= TRAINVAL_WEEK_MAX)]
        wide_trainval = condition_wide(g_trainval)
        static_map = {f: emb_map[(t, f)] for (t, f) in keys if (t, f) in emb_map}
        preds_fam = {}
        if not wide_trainval.empty:
            preds_fam = train_nf_condition_static(
                wide_trainval, static_map, _horizon(), LOOKBACK, best_model, epochs=DL_EPOCHS
            )
        for (t, f) in keys:
            y_true = test_actual[(t, f)]
            pred = preds_fam.get(f)
            if pred is None or len(pred) != len(y_true):
                pred = np.zeros(len(y_true))
            rows.append(_metrics_row(base_row(t, f), y_true, pred, trainval_sales[(t, f)]))
        return pd.DataFrame(rows)

    if best_model in PANEL_MODELS:
        extra_cols = [f"emb_{i}" for i in range(emb_dim)]
        feat_hybrid = feature_cols + extra_cols
        cond_feat = _attach_embeddings(
            feat_df[(feat_df["type"] == typ) & (feat_df[cluster_col] == cluster)], emb_map
        )
        train_scaled, scale_tv = build_scaled_panel_training(cond_feat, keys, feat_hybrid, TRAINVAL_WEEK_MAX)
        if train_scaled.empty:
            return pd.DataFrame()
        model = _fit_scaled_panel(train_scaled, feat_hybrid, best_model, None)
        preds = scaled_recursive_forecast(
            model, cond_feat, keys, feat_hybrid, scale_tv, TRAINVAL_WEEK_MAX, TEST_WEEKS, _horizon()
        )
        for (t, f) in keys:
            y_true = test_actual[(t, f)]
            pred = preds.get((t, f))
            if pred is None or len(pred) != len(y_true):
                pred = np.zeros(len(y_true))
            rows.append(_metrics_row(base_row(t, f), y_true, pred, trainval_sales[(t, f)]))
        return pd.DataFrame(rows)

    for t, f in keys:
        g = df[(df["type"] == t) & (df["family"] == f)]
        g_trainval = g[g["yearweek"] <= TRAINVAL_WEEK_MAX]
        g_test = g[g["yearweek"].isin(TEST_WEEKS)].sort_values("yearweek")
        if g_test.empty:
            continue
        emb = emb_map[(t, f)]
        pred = forecast_one_series(best_model, g_trainval, g_test, feature_cols, embedding=emb, hybrid=True)
        rows.append(_metrics_row(base_row(t, f), test_actual[(t, f)], pred, trainval_sales[(t, f)]))
    return pd.DataFrame(rows)


def run_phase2_all(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_col: str,
    cluster_scheme: str,
    best_df: pd.DataFrame | None = None,
    emb_cache: dict[str, dict[tuple, np.ndarray]] | None = None,
    fixed_model: str | None = None,
) -> pd.DataFrame:
    """Phase2 grid. fixed_model 지정 시 5type×4cluster 전 조건에 동일 base 사용."""
    parts = []
    tasks: list[tuple] = []
    if fixed_model:
        types = sorted(df["type"].unique())
        conditions = build_conditions(types)
        for row in conditions.itertuples(index=False):
            for emb in EMBEDDING_NAMES:
                tasks.append((row.type, int(row.cluster), fixed_model, emb))
    else:
        if best_df is None:
            raise ValueError("best_df or fixed_model required")
        sub_best = best_df[best_df["cluster_scheme"] == cluster_scheme]
        for row in sub_best.itertuples(index=False):
            for emb in EMBEDDING_NAMES:
                tasks.append((row.type, int(row.cluster), row.best_model, emb))

    for typ, cluster, best_model, emb in tqdm(tasks, desc=f"Phase2 {cluster_scheme}"):
        part = run_phase2_condition(
            df,
            feat_df,
            cluster_scheme,
            cluster_col,
            typ,
            cluster,
            best_model,
            emb,
            emb_cache=emb_cache,
        )
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_phase2(
    phase2_df: pd.DataFrame, rank_metric: str = RANK_METRIC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["cluster_scheme", "type", "cluster", "model", "base_model", "embedding"]
    summary, best = _summarize_by_metric(phase2_df, group_cols, rank_metric, model_col="model")
    best = best.rename(columns={"best_model": "best_hybrid", f"best_{rank_metric}": f"best_{rank_metric}"})
    return summary, best


def pick_final_per_condition(
    phase1_best: pd.DataFrame,
    phase2_best: pd.DataFrame,
    rank_metric: str = RANK_METRIC,
) -> pd.DataFrame:
    """조건별 Phase1 vs Phase2 — rank_metric(기본 MAPE) 기준 최종 모델."""
    rows = []
    for row in phase1_best.itertuples(index=False):
        p1_score = getattr(row, f"best_{rank_metric}")
        p2_row = phase2_best[
            (phase2_best["cluster_scheme"] == row.cluster_scheme)
            & (phase2_best["type"] == row.type)
            & (phase2_best["cluster"] == row.cluster)
        ]
        if p2_row.empty:
            continue
        p2 = p2_row.iloc[0]
        p2_score = p2[f"best_{rank_metric}"]
        if p2_score < p1_score:
            winner, model, score = "Phase2", p2["best_hybrid"], p2_score
            p2_model = p2["best_hybrid"]
        else:
            winner, model, score = "Phase1", row.best_model, p1_score
            p2_model = p2["best_hybrid"]
        rows.append(
            {
                "cluster_scheme": row.cluster_scheme,
                "type": row.type,
                "cluster": row.cluster,
                "winner": winner,
                "final_model": model,
                rank_metric: score,
                "phase1_best": row.best_model,
                f"phase1_{rank_metric}": p1_score,
                "phase2_best": p2_model,
                f"phase2_{rank_metric}": p2_score,
            }
        )
    return pd.DataFrame(rows)
