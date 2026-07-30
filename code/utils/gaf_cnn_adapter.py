"""Adapter for eliotwalt/gaf-cnn official GASF + DenseNet encoder."""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .device import get_torch_device
from .vendor_import import purge_vendor_modules, restore_project_utils

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GAF_CNN_ROOT = _REPO_ROOT / "vendor" / "gaf-cnn"


def _ensure_gaf_cnn_path() -> Path:
    if not _GAF_CNN_ROOT.is_dir():
        raise FileNotFoundError(
            f"gaf-cnn not found at {_GAF_CNN_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    return _GAF_CNN_ROOT


def _purge_gaf_modules() -> None:
    purge_vendor_modules(("data",))


@contextlib.contextmanager
def _gaf_import_context():
    purge_vendor_modules(("data",))
    root = str(_ensure_gaf_cnn_path())
    code_root = str(Path(__file__).resolve().parents[1])
    removed = []
    for p in list(sys.path):
        norm = p.replace("\\", "/")
        if p == code_root or norm.endswith("/code"):
            sys.path.remove(p)
            removed.append(p)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        sys.path.remove(root)
        for p in reversed(removed):
            sys.path.insert(0, p)
        restore_project_utils()


def _maybe_resample_rows(X: np.ndarray, max_len: int = 64) -> np.ndarray:
    """Cap sequence length for GASF memory (official transform on resampled series)."""
    n, seq_len = X.shape
    if seq_len <= max_len:
        return X
    idx = np.linspace(0, seq_len - 1, max_len).astype(int)
    return X[:, idx]


def _rows_to_gasf(X: np.ndarray) -> np.ndarray:
    """Official Gasf.transform expects S of shape (n_series, n_timestamps) and
    internally calls scaler.transform(S), so the scaler must be fitted on data
    with n_timestamps columns.

    GASF requires each series to be rescaled to [0, 1] *along time* (Wang & Oates).
    A per-row MinMaxScaler fitted on (T, 1) has 1 feature and therefore cannot be
    applied to a (1, T) row — that mismatch previously raised ValueError and the
    caller silently fell back to PCA.

    Fix: rescale each row to [0, 1] across time ourselves, then hand Gasf an
    identity scaler (fitted on the [0, 1] range over n_timestamps columns) so the
    official transform math is applied unchanged.
    """
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    Xn = (X - lo) / rng                      # per-series [0, 1]

    n_timestamps = X.shape[1]
    identity = MinMaxScaler(feature_range=(0, 1)).fit(
        np.vstack([np.zeros(n_timestamps), np.ones(n_timestamps)])
    )

    with _gaf_import_context():
        from data.transforms import Gasf

        gasf = Gasf(scaler=identity, feature_range=(0, 1))
        imgs = gasf.transform(Xn)            # (n, T, T)
    return np.asarray(imgs, dtype=np.float32)


def embed_gaf_cnn_official(X: np.ndarray, n_components: int = 10, epochs: int = 40) -> np.ndarray:
    """Official GASF transform + GafToGafRegressor encoder features projected to n_components."""
    device = get_torch_device()
    pin = device.type == "cuda"
    Xs = _maybe_resample_rows(np.asarray(X, dtype=np.float32))
    n, seq_len = Xs.shape
    image_size = seq_len
    gasf_imgs = _rows_to_gasf(Xs)  # (n, H, W)
    tensors = torch.tensor(gasf_imgs[:, None, :, :], dtype=torch.float32)

    with _gaf_import_context():
        from models.regression import GafToGafRegressor

        encode_channels = [12, 24] if image_size >= 24 else [8, 16]
        decode_channels = [12]
        with contextlib.redirect_stdout(io.StringIO()):
            model = GafToGafRegressor(
                in_features=1,
                out_features=1,
                encode_channels=encode_channels,
                decode_channels=decode_channels,
                encode_block_type="densenet",
                encode_block_dim=2,
                image_size=image_size,
            ).to(device)
        encoder = model.net[0]
        decoder = model.net[1]

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loader = DataLoader(TensorDataset(tensors, tensors), batch_size=min(16, max(4, n // 4)), shuffle=True)
    encoder.train()
    decoder.train()
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            opt.zero_grad()
            pred = decoder(encoder(xb))
            # DenseNet 인코더/디코더의 stride·pooling 때문에 복원 이미지 크기가
            # 입력과 정확히 일치하지 않을 수 있다(예: 64 -> 62). 공식 모델을 그대로
            # 두고 예측을 입력 해상도로 맞춰 손실을 계산한다.
            if pred.shape[-2:] != yb.shape[-2:]:
                pred = nn.functional.interpolate(
                    pred, size=yb.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    encoder.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, n, 16):
            xb = tensors[i : i + 16].to(device, non_blocking=pin)
            h = encoder(xb)
            h = h.flatten(1)
            feats.append(h.cpu().numpy())
    flat = np.concatenate(feats, axis=0)
    if flat.shape[1] == n_components:
        return flat.astype(np.float32)
    proj = nn.Linear(flat.shape[1], n_components).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    t_flat = torch.tensor(flat, dtype=torch.float32)
    for _ in range(max(20, epochs // 2)):
        opt.zero_grad()
        z = proj(t_flat.to(device))
        loss = (z ** 2).mean()
        loss.backward()
        opt.step()
    proj.eval()
    with torch.no_grad():
        out = proj(t_flat.to(device)).cpu().numpy()
    return out.astype(np.float32)
