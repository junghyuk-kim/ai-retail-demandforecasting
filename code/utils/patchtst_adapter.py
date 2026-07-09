"""Adapter for PatchTST/PatchTST official self-supervised PatchTST backbone."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .device import get_torch_device
from .vendor_import import restore_project_utils

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATCHTST_ROOT = _REPO_ROOT / "vendor" / "PatchTST" / "PatchTST_self_supervised"


def _ensure_patchtst_path() -> Path:
    if not _PATCHTST_ROOT.is_dir():
        raise FileNotFoundError(
            f"PatchTST not found at {_PATCHTST_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    return _PATCHTST_ROOT


def _series_to_patches(X: np.ndarray, patch_len: int, stride: int) -> tuple[np.ndarray, int]:
    n, seq_len = X.shape
    patch_len = min(patch_len, seq_len)
    stride = max(1, min(stride, patch_len))
    n_patches = max(1, (seq_len - patch_len) // stride + 1)
    patches = np.zeros((n, n_patches, 1, patch_len), dtype=np.float32)
    for i in range(n):
        for j in range(n_patches):
            start = j * stride
            patches[i, j, 0, :] = X[i, start : start + patch_len]
    return patches, n_patches


def embed_patchtst_official(X: np.ndarray, n_components: int = 10, epochs: int = 60) -> np.ndarray:
    """Train PatchTST regression head; return official backbone pooled embeddings."""
    device = get_torch_device()
    pin = device.type == "cuda"
    Xs = np.asarray(X, dtype=np.float32)
    n, seq_len = Xs.shape
    patch_len = min(8, max(2, seq_len // 4))
    stride = max(1, patch_len // 2)
    patches, n_patches = _series_to_patches(Xs, patch_len, stride)
    targets = Xs.mean(axis=1, keepdims=True).astype(np.float32)

    patchtst_root = str(_ensure_patchtst_path())
    inserted = False
    if patchtst_root not in sys.path:
        sys.path.insert(0, patchtst_root)
        inserted = True
    try:
        from src.models.patchTST import PatchTST
    finally:
        if inserted:
            sys.path.remove(patchtst_root)
        restore_project_utils()

    n_heads = 4 if n_components % 4 == 0 else (2 if n_components % 2 == 0 else 1)
    model = PatchTST(
        c_in=1,
        target_dim=1,
        patch_len=patch_len,
        stride=stride,
        num_patch=n_patches,
        n_layers=3,
        d_model=n_components,
        n_heads=n_heads,
        d_ff=max(64, n_components * 4),
        dropout=0.1,
        head_dropout=0.0,
        head_type="regression",
        individual=False,
    ).to(device)

    t_x = torch.tensor(patches)
    t_y = torch.tensor(targets)
    loader = DataLoader(
        TensorDataset(t_x, t_y),
        batch_size=min(32, max(4, n // 4)),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            opt.zero_grad()
            pred = model(xb)
            if pred.ndim == 3:
                pred = pred.squeeze(-1)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, n, 32):
            xb = torch.tensor(patches[i : i + 32]).to(device, non_blocking=pin)
            z = model.backbone(xb)
            z_pool = z.mean(dim=-1).reshape(z.size(0), -1)
            embs.append(z_pool.cpu().numpy())
    out = np.concatenate(embs, axis=0).astype(np.float32)
    if out.shape[1] > n_components:
        out = out[:, :n_components]
    return out
