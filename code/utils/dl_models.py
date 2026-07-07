"""Deep learning forecasters: LSTM, Autoformer, N-HiTS, iTransformer (CPU)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_windows(series: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class NHITSBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x):
        return self.net(x.squeeze(-1))


class AutoformerLite(nn.Module):
    def __init__(self, lookback: int, horizon: int, d_model: int = 64):
        super().__init__()
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Linear(lookback, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        s = x.squeeze(-1)
        trend = self.decomp(s.unsqueeze(1)).squeeze(1)
        resid = s - trend
        h = self.proj(resid).unsqueeze(1)
        h = self.encoder(h).squeeze(1)
        return self.fc(h)


class ITransformerLite(nn.Module):
    def __init__(self, lookback: int, horizon: int, d_model: int = 64):
        super().__init__()
        self.var_proj = nn.Linear(lookback, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        v = self.var_proj(x.squeeze(-1)).unsqueeze(1)
        h = self.encoder(v).squeeze(1)
        return self.fc(h)


def train_dl_forecaster(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    model_cls,
    epochs: int = 100,
    lr: float = 1e-3,
) -> np.ndarray:
    device = torch.device("cpu")
    y = series.astype(np.float32)
    if len(y) < lookback + horizon + 5:
        return np.full(horizon, max(float(y[-1]), 0.0))
    X, Y = _make_windows(y, lookback, horizon)
    if len(X) < 5:
        return np.full(horizon, max(float(y[-1]), 0.0))
    X_t = torch.tensor(X).unsqueeze(-1)
    Y_t = torch.tensor(Y)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = model_cls(lookback, horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        last = torch.tensor(y[-lookback:]).float().unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(last).cpu().numpy().reshape(-1)
    return np.maximum(pred, 0.0)


DL_MODELS = {
    "LSTM": lambda lb, h, s: train_dl_forecaster(s, lb, h, LSTMForecaster, epochs=100),
    "N-HiTS": lambda lb, h, s: train_dl_forecaster(s, lb, h, NHITSBlock, epochs=100),
    "Autoformer": lambda lb, h, s: train_dl_forecaster(s, lb, h, AutoformerLite, epochs=100),
    "iTransformer": lambda lb, h, s: train_dl_forecaster(s, lb, h, ITransformerLite, epochs=100),
}
