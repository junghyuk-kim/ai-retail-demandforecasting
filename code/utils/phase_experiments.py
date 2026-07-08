"""Phase 1 / Phase 2 forecasting experiments (40 conditions x 10 algos)."""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .dl_models import train_dl_forecaster
from .embeddings import EMBEDDERS
from .forecasting import (
    FORECASTERS,
    forecast_arima,
    forecast_ml_panel,
    forecast_prophet,
)
from .intermittent import forecast_sba, forecast_tsb
from .metrics import FORECAST_METRICS, forecast_metrics, wmape
from .splits import TRAIN_WEEK_MAX, VAL_WEEKS

STAT_MODELS = ["ARIMA", "Prophet", "SBA", "TSB"]
ML_MODELS = ["RF", "XGBoost"]
DL_MODELS = ["LSTM", "Autoformer", "N-HiTS", "iTransformer"]

PHASE1_MODELS = STAT_MODELS + ML_MODELS + DL_MODELS
RANK_METRIC = "mape"  # 조건별 Best 선정 기준 (MAE/RMSE/MAPE/MASE 중)

EMBEDDING_NAMES = list(EMBEDDERS.keys())
LOOKBACK = 16
try:
    import torch

    DL_EPOCHS = 80 if torch.cuda.is_available() else 40
except Exception:
    DL_EPOCHS = 40
N_EMB_DIM = 10

SERIES_MODELS = {"ARIMA", "Prophet", "SBA", "TSB", "LSTM", "Autoformer", "N-HiTS", "iTransformer"}
PANEL_MODELS = {"RF", "XGBoost"}


def _horizon() -> int:
    return len(VAL_WEEKS)


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

    if model in {"LSTM", "Autoformer", "N-HiTS", "iTransformer"}:
        from .dl_models import (
            AutoformerLite,
            ITransformerLite,
            LSTMForecaster,
            NHITSBlock,
            train_dl_forecaster,
        )

        cls_map = {
            "LSTM": LSTMForecaster,
            "Autoformer": AutoformerLite,
            "N-HiTS": NHITSBlock,
            "iTransformer": ITransformerLite,
        }
        pred = train_dl_forecaster(
            sales.astype(float).values, lookback, horizon, cls_map[model], epochs=DL_EPOCHS
        )
        if hybrid and embedding is not None:
            scale = 1.0 + 0.05 * np.tanh(float(embedding.mean()))
            pred = pred * scale
        return pred

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


def run_phase1_condition(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_scheme: str,
    cluster_col: str,
    typ: str,
    cluster: int,
    models: list[str] | None = None,
) -> pd.DataFrame:
    models = models or PHASE1_MODELS
    feature_cols = _feature_cols(feat_df)
    horizon = _horizon()
    rows = []

    series_keys = (
        df[(df["type"] == typ) & (df[cluster_col] == cluster)][["type", "family"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    series_list = list(series_keys)
    if not series_list:
        return pd.DataFrame()

    panel_models = [m for m in models if m in PANEL_MODELS]
    series_models = [m for m in models if m in SERIES_MODELS]

    # Panel ML: train once per condition
    panel_preds: dict[str, dict[tuple, np.ndarray]] = {m: {} for m in panel_models}
    if panel_models:
        keys_set = set(series_list)
        train_cond = feat_df[
            (feat_df["type"] == typ)
            & (feat_df[cluster_col] == cluster)
            & (feat_df["yearweek"] <= TRAIN_WEEK_MAX)
        ].dropna(subset=feature_cols)
        val_cond = feat_df[
            (feat_df["type"] == typ)
            & (feat_df[cluster_col] == cluster)
            & (feat_df["yearweek"].isin(VAL_WEEKS))
        ].copy()
        for pm in panel_models:
            if train_cond.empty or val_cond.empty:
                continue
            pred_all = forecast_ml_panel(
                train_cond,
                val_cond,
                feature_cols,
                model_name="rf" if pm == "RF" else "xgboost",
            )
            val_cond = val_cond.copy()
            val_cond["pred"] = pred_all
            for (t, f), g in val_cond.groupby(["type", "family"]):
                if (t, f) in keys_set:
                    panel_preds[pm][(t, f)] = g.sort_values("yearweek")["pred"].values

    for typ2, fam in series_list:
        g = df[(df["type"] == typ2) & (df["family"] == fam)]
        g_train = g[g["yearweek"] <= TRAIN_WEEK_MAX]
        g_val = g[g["yearweek"].isin(VAL_WEEKS)].sort_values("yearweek")
        if g_val.empty:
            continue
        y_true = g_val["sales"].values
        y_train = g_train["sales"].values

        for model in series_models:
            pred = forecast_one_series(model, g_train, g_val, feature_cols)
            rows.append(
                _metrics_row(
                    {
                        "phase": 1,
                        "cluster_scheme": cluster_scheme,
                        "type": typ2,
                        "cluster": cluster,
                        "family": fam,
                        "model": model,
                    },
                    y_true,
                    pred,
                    y_train,
                )
            )

        for model in panel_models:
            pred = panel_preds.get(model, {}).get((typ2, fam))
            if pred is None or len(pred) != len(y_true):
                pred = np.zeros(len(y_true))
            rows.append(
                _metrics_row(
                    {
                        "phase": 1,
                        "cluster_scheme": cluster_scheme,
                        "type": typ2,
                        "cluster": cluster,
                        "family": fam,
                        "model": model,
                    },
                    y_true,
                    pred,
                    y_train,
                )
            )

    return pd.DataFrame(rows)


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
        s = df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAIN_WEEK_MAX)]["sales"].values
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
            s = df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAIN_WEEK_MAX)]["sales"].values
            train_rows.append(s)
            keys.append((t, f))
        train_matrix = np.vstack(train_rows).astype(float)
        emb_all = _fit_embedding(train_matrix, EMBEDDERS[embedding_name])
        emb_map = {keys[i]: emb_all[i] for i in range(len(keys))}

    combo = f"{best_model}+{embedding_name}"
    emb_dim = next(iter(emb_map.values())).shape[0]

    if best_model in PANEL_MODELS:
        extra_cols = [f"emb_{i}" for i in range(emb_dim)]
        train_cond = feat_df[
            (feat_df["type"] == typ)
            & (feat_df[cluster_col] == cluster)
            & (feat_df["yearweek"] <= TRAIN_WEEK_MAX)
        ].dropna(subset=feature_cols)
        val_cond = feat_df[
            (feat_df["type"] == typ)
            & (feat_df[cluster_col] == cluster)
            & (feat_df["yearweek"].isin(VAL_WEEKS))
        ].copy()
        train_cond = _attach_embeddings(train_cond, emb_map)
        val_cond = _attach_embeddings(val_cond, emb_map)
        feat_hybrid = feature_cols + extra_cols
        pred_all = forecast_ml_panel(
            train_cond,
            val_cond,
            feat_hybrid,
            model_name="rf" if best_model == "RF" else "xgboost",
        )
        val_cond["pred"] = pred_all
        for (t, f), g in val_cond.groupby(["type", "family"]):
            g_val = g.sort_values("yearweek")
            y_true = g_val["sales"].values
            g_train = df[(df["type"] == t) & (df["family"] == f) & (df["yearweek"] <= TRAIN_WEEK_MAX)]
            y_train = g_train["sales"].values
            rows.append(
                _metrics_row(
                    {
                        "phase": 2,
                        "cluster_scheme": cluster_scheme,
                        "type": t,
                        "cluster": cluster,
                        "family": f,
                        "model": combo,
                        "base_model": best_model,
                        "embedding": embedding_name,
                    },
                    y_true,
                    g_val["pred"].values,
                    y_train,
                )
            )
        return pd.DataFrame(rows)

    for t, f in series_list:
        g = df[(df["type"] == t) & (df["family"] == f)]
        g_train = g[g["yearweek"] <= TRAIN_WEEK_MAX]
        g_val = g[g["yearweek"].isin(VAL_WEEKS)].sort_values("yearweek")
        if g_val.empty:
            continue
        y_true = g_val["sales"].values
        y_train = g_train["sales"].values
        emb = emb_map[(t, f)]
        pred = forecast_one_series(
            best_model, g_train, g_val, feature_cols, embedding=emb, hybrid=True
        )
        rows.append(
            _metrics_row(
                {
                    "phase": 2,
                    "cluster_scheme": cluster_scheme,
                    "type": t,
                    "cluster": cluster,
                    "family": f,
                    "model": combo,
                    "base_model": best_model,
                    "embedding": embedding_name,
                },
                y_true,
                pred,
                y_train,
            )
        )
    return pd.DataFrame(rows)


def run_phase2_all(
    df: pd.DataFrame,
    feat_df: pd.DataFrame,
    cluster_col: str,
    cluster_scheme: str,
    best_df: pd.DataFrame,
    emb_cache: dict[str, dict[tuple, np.ndarray]] | None = None,
) -> pd.DataFrame:
    parts = []
    sub_best = best_df[best_df["cluster_scheme"] == cluster_scheme]
    tasks = []
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
