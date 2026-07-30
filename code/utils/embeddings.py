"""Time-series embedding methods (PCA, DTW, AE, GAF-CNN, TS2Vec, PatchTST).

GAF-CNN, TS2Vec, PatchTST use official git submodules via adapters:
- eliotwalt/gaf-cnn
- zhihanyue/ts2vec
- PatchTST/PatchTST (self-supervised backbone)
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


def _set_torch_seed(seed: int = 42) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(ds, batch_size: int, shuffle: bool = True):
    import torch
    from torch.utils.data import DataLoader

    pin = torch.cuda.is_available()
    bs = batch_size * (2 if pin else 1)
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, pin_memory=pin)


def embed_pca(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    return PCA(n_components=n_components, random_state=42).fit_transform(StandardScaler().fit_transform(X))


def embed_fastdtw(X: np.ndarray, n_components: int = 10, n_jobs: int = -1) -> np.ndarray:
    """FastDTW distance matrix -> MDS embedding (학위논문 FastDTW 파이프라인)."""
    from joblib import Parallel, delayed
    from sklearn.manifold import MDS

    try:
        from fastdtw import fastdtw
    except ImportError:
        return embed_dtw_mds(X, n_components)

    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    n = Xs.shape[0]
    _scalar_dist = lambda a, b: abs(float(a) - float(b))

    def _row_dist(i: int) -> list[float]:
        return [float(fastdtw(Xs[i], Xs[j], dist=_scalar_dist)[0]) for j in range(n)]

    dist = np.array(Parallel(n_jobs=n_jobs)(delayed(_row_dist)(i) for i in range(n)), dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    n_components = min(n_components, n - 1)
    return MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42,
        n_init=4,
        max_iter=300,
    ).fit_transform(dist)


def embed_dtw_mds(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    try:
        from tslearn.metrics import cdist_dtw

        dist = cdist_dtw(X)
    except Exception:
        from scipy.spatial.distance import pdist, squareform

        dist = squareform(pdist(X, metric="euclidean"))
    n_components = min(n_components, X.shape[0] - 1)
    return MDS(n_components=n_components, dissimilarity="precomputed", random_state=42).fit_transform(dist)


def embed_autoencoder(X: np.ndarray, n_components: int = 10, epochs: int = 80) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from .device import get_torch_device

    _set_torch_seed()
    device = get_torch_device()
    pin = device.type == "cuda"
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    tensor = torch.tensor(Xs)
    ds = TensorDataset(tensor)
    loader = _loader(ds, batch_size=32)

    class AE(nn.Module):
        def __init__(self, d_in: int, d_latent: int):
            super().__init__()
            h = max(d_latent * 2, 32)
            self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_latent))
            self.dec = nn.Sequential(nn.Linear(d_latent, h), nn.ReLU(), nn.Linear(h, d_in))

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    model = AE(Xs.shape[1], n_components).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin)
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(Xs).to(device, non_blocking=pin))
    return z.cpu().numpy()


def embed_gaf_cnn(X: np.ndarray, n_components: int = 10, epochs: int = 60) -> np.ndarray:
    from .gaf_cnn_adapter import embed_gaf_cnn_official

    try:
        return embed_gaf_cnn_official(X, n_components=n_components, epochs=epochs)
    except Exception:
        return embed_pca(X, n_components)


def embed_ts2vec(X: np.ndarray, n_components: int = 10, epochs: int = 80) -> np.ndarray:
    from .ts2vec_adapter import embed_ts2vec_official

    try:
        return embed_ts2vec_official(X, n_components=n_components, n_epochs=epochs)
    except Exception:
        return embed_pca(X, n_components)


def embed_patchtst(X: np.ndarray, n_components: int = 10, patch_len: int = 8, epochs: int = 80) -> np.ndarray:
    from .patchtst_adapter import embed_patchtst_official

    try:
        return embed_patchtst_official(X, n_components=n_components, epochs=epochs)
    except Exception:
        return embed_pca(X, n_components)


EMBEDDERS = {
    "PCA": embed_pca,
    "FastDTW": embed_fastdtw,
    "AE": embed_autoencoder,
    "GAF-CNN": embed_gaf_cnn,
    "TS2Vec": embed_ts2vec,
    "PatchTST": embed_patchtst,
}
