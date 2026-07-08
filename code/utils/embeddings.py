"""Time-series embedding methods (PCA, DTW, AE, GAF-CNN, TS2Vec, PatchTST)."""
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
    """FastDTW distance matrix -> MDS embedding (Daiso FastDTW 파이프라인)."""
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
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    _set_torch_seed()

    try:
        from pyts.image import GramianAngularField
    except Exception:
        return embed_pca(X, n_components)

    gaf = GramianAngularField(image_size=min(32, X.shape[1]))
    imgs = gaf.fit_transform(StandardScaler().fit_transform(X)).astype(np.float32)
    imgs = imgs[:, None, :, :]
    from .device import get_torch_device

    device = get_torch_device()
    pin = device.type == "cuda"

    class GAFCNN(nn.Module):
        def __init__(self, latent: int):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(32, latent)

        def forward(self, x):
            h = self.cnn(x).view(x.size(0), -1)
            return self.fc(h)

    model = GAFCNN(n_components).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.tensor(imgs))
    loader = _loader(ds, batch_size=16)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin)
            opt.zero_grad()
            z = model(batch)
            loss = (z ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(imgs).to(device, non_blocking=pin))
    return z.cpu().numpy()


def embed_ts2vec(X: np.ndarray, n_components: int = 10, epochs: int = 80) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from .device import get_torch_device

    _set_torch_seed()
    device = get_torch_device()
    pin = device.type == "cuda"
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    t = torch.tensor(Xs)

    class Encoder(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, d_out),
            )

        def forward(self, x):
            return self.net(x)

    enc = Encoder(Xs.shape[1], n_components).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    ds = TensorDataset(t)
    loader = _loader(ds, batch_size=32)
    enc.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin)
            noise = batch + 0.05 * torch.randn_like(batch)
            opt.zero_grad()
            z1 = enc(batch)
            z2 = enc(noise)
            loss = ((z1 - z2) ** 2).mean()
            loss.backward()
            opt.step()
    enc.eval()
    with torch.no_grad():
        return enc(t.to(device, non_blocking=pin)).cpu().numpy()


def embed_patchtst(X: np.ndarray, n_components: int = 10, patch_len: int = 8, epochs: int = 80) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from .device import get_torch_device

    _set_torch_seed()
    device = get_torch_device()
    pin = device.type == "cuda"
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    seq_len = Xs.shape[1]
    patch_len = min(patch_len, seq_len)
    n_patches = seq_len // patch_len
    if n_patches < 1:
        return embed_pca(X, n_components)
    Xp = Xs[:, : n_patches * patch_len].reshape(Xs.shape[0], n_patches, patch_len)
    t = torch.tensor(Xp)

    class PatchTSTEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(patch_len, n_components)
            nhead = 2 if n_components % 2 == 0 else 1
            enc_layer = nn.TransformerEncoderLayer(
                d_model=n_components, nhead=nhead, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        def forward(self, x):
            h = self.proj(x)
            h = self.encoder(h)
            return h.mean(dim=1)

    model = PatchTSTEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(t)
    loader = _loader(ds, batch_size=32)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin)
            opt.zero_grad()
            z = model(batch)
            loss = (z ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(t.to(device, non_blocking=pin)).cpu().numpy()


EMBEDDERS = {
    "PCA": embed_pca,
    "FastDTW": embed_fastdtw,
    "AE": embed_autoencoder,
    "GAF-CNN": embed_gaf_cnn,
    "TS2Vec": embed_ts2vec,
    "PatchTST": embed_patchtst,
}
