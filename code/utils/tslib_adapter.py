"""Adapter for thuml/Time-Series-Library Autoformer & iTransformer.

Uses Model classes from vendor/Time-Series-Library (git submodule).
Not the standalone thuml/Autoformer or thuml/iTransformer run.py pipelines.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .device import get_torch_device

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TSLIB_ROOT = _REPO_ROOT / "vendor" / "Time-Series-Library"


def _ensure_tslib_path() -> Path:
    if not _TSLIB_ROOT.is_dir():
        raise FileNotFoundError(
            f"Time-Series-Library not found at {_TSLIB_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    return _TSLIB_ROOT


def _purge_tslib_modules() -> None:
    """Drop cached imports that shadow thuml Time-Series-Library packages."""
    prefixes = ("utils", "models", "layers")
    for key in list(sys.modules):
        if key in prefixes or any(key.startswith(f"{p}.") for p in prefixes):
            del sys.modules[key]


@contextlib.contextmanager
def _tslib_import_context():
    """Avoid shadowing tslib's `utils` package by our `code/utils`."""
    _purge_tslib_modules()
    tslib_root = str(_ensure_tslib_path())
    code_root = str(Path(__file__).resolve().parents[1])
    removed = []
    for p in list(sys.path):
        norm = p.replace("\\", "/")
        if p == code_root or norm.endswith("/code"):
            sys.path.remove(p)
            removed.append(p)
    sys.path.insert(0, tslib_root)
    try:
        yield
    finally:
        sys.path.remove(tslib_root)
        for p in reversed(removed):
            sys.path.insert(0, p)


def _load_model(model_name: str, configs: SimpleNamespace) -> nn.Module:
    import importlib

    with _tslib_import_context():
        if model_name == "Autoformer":
            mod = importlib.import_module("models.Autoformer")
            return mod.Model(configs)
        if model_name == "iTransformer":
            mod = importlib.import_module("models.iTransformer")
            return mod.Model(configs)
    raise ValueError(f"Unsupported tslib model: {model_name}")


def _make_config(seq_len: int, pred_len: int, label_len: int, enc_in: int = 1) -> SimpleNamespace:
    moving_avg = min(25, max(3, seq_len // 3))
    if moving_avg % 2 == 0:
        moving_avg += 1
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        moving_avg=moving_avg,
        factor=3,
        dropout=0.1,
        embed="fixed",
        freq="w",
        activation="gelu",
        num_class=0,
    )


def _make_windows(
    series: np.ndarray,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """TSlib-style encoder/decoder windows: X [N,L,C], Y [N,label_len+pred_len,C]."""
    y = series.astype(np.float32)
    xs, ys = [], []
    total_y = label_len + pred_len
    for i in range(len(y) - seq_len - pred_len + 1):
        s_end = i + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + total_y
        xs.append(y[i:s_end])
        ys.append(y[r_begin:r_end])
    if not xs:
        return np.empty((0, seq_len, 1), dtype=np.float32), np.empty((0, total_y, 1), dtype=np.float32)
    return (
        np.array(xs, dtype=np.float32)[:, :, None],
        np.array(ys, dtype=np.float32)[:, :, None],
    )


def _predict(
    model: nn.Module,
    model_name: str,
    series: np.ndarray,
    seq_len: int,
    label_len: int,
    pred_len: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x_enc = torch.tensor(series[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1).to(device)
    with torch.no_grad():
        if model_name == "iTransformer":
            out = model(x_enc, None, None, None)
            pred = out[:, -pred_len:, 0].cpu().numpy()
        else:
            known = torch.tensor(series[-label_len:], dtype=torch.float32).reshape(1, label_len, 1).to(device)
            dec_zeros = torch.zeros(1, pred_len, 1, device=device)
            dec_inp = torch.cat([known, dec_zeros], dim=1)
            out = model(x_enc, None, dec_inp, None)
            pred = out[:, -pred_len:, 0].cpu().numpy()
    return pred.reshape(-1)


def _make_windows_mv(
    wide: np.ndarray,
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate windows: X [N,L,C], Y [N,label_len+pred_len,C]."""
    mat = wide.astype(np.float32)
    t_steps, _ = mat.shape
    xs, ys = [], []
    total_y = label_len + pred_len
    for i in range(t_steps - seq_len - pred_len + 1):
        s_end = i + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + total_y
        xs.append(mat[i:s_end, :])
        ys.append(mat[r_begin:r_end, :])
    if not xs:
        return (
            np.empty((0, seq_len, mat.shape[1]), dtype=np.float32),
            np.empty((0, total_y, mat.shape[1]), dtype=np.float32),
        )
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _predict_mv(
    model: nn.Module,
    model_name: str,
    wide: np.ndarray,
    seq_len: int,
    label_len: int,
    pred_len: int,
    device: torch.device,
) -> np.ndarray:
    """Return [pred_len, n_variates]."""
    model.eval()
    x_enc = torch.tensor(wide[-seq_len:, :], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        if model_name == "iTransformer":
            out = model(x_enc, None, None, None)
        else:
            known = torch.tensor(wide[-label_len:, :], dtype=torch.float32).unsqueeze(0).to(device)
            dec_zeros = torch.zeros(1, pred_len, wide.shape[1], device=device)
            dec_inp = torch.cat([known, dec_zeros], dim=1)
            out = model(x_enc, None, dec_inp, None)
    return out[:, -pred_len:, :].cpu().numpy()[0]


def train_tslib_condition(
    wide_train: pd.DataFrame,
    horizon: int,
    lookback: int,
    model_name: str,
    epochs: int = 40,
    label_len: int | None = None,
) -> dict[str, np.ndarray]:
    """조건 내 다변량 패널 — iTransformer/Autoformer 공식 구현."""
    mat = wide_train.fillna(0).to_numpy(dtype=np.float32)
    families = list(wide_train.columns)
    seq_len = lookback
    pred_len = horizon
    label_len = label_len or max(1, min(seq_len // 2, pred_len * 2))

    if mat.shape[0] < seq_len + pred_len + 5 or mat.shape[1] == 0:
        return {}

    X, Y = _make_windows_mv(mat, seq_len, label_len, pred_len)
    if len(X) < 3:
        return {}

    device = get_torch_device()
    configs = _make_config(seq_len, pred_len, label_len, enc_in=mat.shape[1])
    model = _load_model(model_name, configs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    pin = device.type == "cuda"
    batch_size = min(128 if pin else 32, len(X))
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(Y)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=pin)
            batch_y = batch_y.to(device, non_blocking=pin)
            opt.zero_grad()
            if model_name == "iTransformer":
                out = model(batch_x, None, None, None)
            else:
                dec_zeros = torch.zeros(batch_x.size(0), pred_len, mat.shape[1], device=device)
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1)
                out = model(batch_x, None, dec_inp, None)
            loss = loss_fn(out[:, -pred_len:, :], batch_y[:, -pred_len:, :])
            loss.backward()
            opt.step()

    pred_mat = _predict_mv(model, model_name, mat, seq_len, label_len, pred_len, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {fam: np.maximum(pred_mat[:, i], 0.0) for i, fam in enumerate(families)}


def train_tslib_forecaster(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    model_name: str,
    epochs: int = 40,
    label_len: int | None = None,
) -> np.ndarray:
    """Train official thuml Autoformer/iTransformer on one univariate series."""
    device = get_torch_device()
    y = np.asarray(series, dtype=np.float32)
    seq_len = lookback
    pred_len = horizon
    label_len = label_len or max(1, min(seq_len // 2, pred_len * 2))

    if len(y) < seq_len + pred_len + 5:
        return np.full(pred_len, max(float(y[-1]) if len(y) else 0.0, 0.0))

    X, Y = _make_windows(y, seq_len, label_len, pred_len)
    if len(X) < 3:
        return np.full(pred_len, max(float(y[-1]), 0.0))

    configs = _make_config(seq_len, pred_len, label_len, enc_in=1)
    model = _load_model(model_name, configs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    pin = device.type == "cuda"
    batch_size = min(128 if pin else 32, len(X))
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(Y)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=pin)
            batch_y = batch_y.to(device, non_blocking=pin)
            opt.zero_grad()
            if model_name == "iTransformer":
                out = model(batch_x, None, None, None)
            else:
                dec_zeros = torch.zeros(batch_x.size(0), pred_len, 1, device=device)
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1)
                out = model(batch_x, None, dec_inp, None)
            pred = out[:, -pred_len:, :]
            target = batch_y[:, -pred_len:, :]
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()

    pred = _predict(model, model_name, y, seq_len, label_len, pred_len, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.maximum(pred.astype(float), 0.0)
