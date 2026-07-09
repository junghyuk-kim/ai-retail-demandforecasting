"""Adapter for zhihanyue/ts2vec official TS2Vec representation learning."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np

from .vendor_import import purge_vendor_modules, restore_project_utils

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TS2VEC_ROOT = _REPO_ROOT / "vendor" / "ts2vec"


def _ensure_ts2vec_path() -> Path:
    if not _TS2VEC_ROOT.is_dir():
        raise FileNotFoundError(
            f"TS2Vec not found at {_TS2VEC_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    return _TS2VEC_ROOT


@contextlib.contextmanager
def _ts2vec_import_context():
    purge_vendor_modules()
    ts2vec_root = str(_ensure_ts2vec_path())
    code_root = str(Path(__file__).resolve().parents[1])
    removed = []
    for p in list(sys.path):
        norm = p.replace("\\", "/")
        if p == code_root or norm.endswith("/code"):
            sys.path.remove(p)
            removed.append(p)
    sys.path.insert(0, ts2vec_root)
    try:
        yield
    finally:
        sys.path.remove(ts2vec_root)
        for p in reversed(removed):
            sys.path.insert(0, p)
        restore_project_utils()


def embed_ts2vec_official(X: np.ndarray, n_components: int = 10, n_epochs: int | None = None) -> np.ndarray:
    """Train official TS2Vec on panel rows and return series-level embeddings."""
    from .device import get_torch_device

    n, seq_len = X.shape
    data = np.asarray(X, dtype=np.float32)[:, :, None]  # (n, T, 1)
    device = str(get_torch_device())
    batch_size = min(16, max(n // 2, 2))

    with _ts2vec_import_context():
        from ts2vec import TS2Vec

        if n_epochs is None:
            n_iters = 120 if data.size <= 100_000 else 300
        else:
            n_iters = max(n_epochs * 3, 40)

        depth = 6 if seq_len < 64 else 10
        model = TS2Vec(
            input_dims=1,
            output_dims=n_components,
            hidden_dims=64,
            depth=depth,
            device=device,
            batch_size=batch_size,
            max_train_length=min(64, seq_len) if seq_len > 32 else None,
        )
        try:
            model.fit(data, n_iters=n_iters, verbose=False)
        except Exception:
            model.fit(data, n_epochs=max(10, (n_iters // 3)), verbose=False)
        repr_ = model.encode(data, encoding_window="full_series")
    out = np.asarray(repr_, dtype=np.float32)
    if out.ndim == 3:
        out = out.squeeze(1)
    if out.shape[1] > n_components:
        out = out[:, :n_components]
    return out
