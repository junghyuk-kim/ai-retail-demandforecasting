"""Syntetos-Boylan Classification (SBC) utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

ADI_TH = 1.32
CV2_TH = 0.49

CLUSTER_LABELS = {
    1: "Smooth",
    2: "Intermittent",
    3: "Erratic",
    4: "Lumpy",
}


def assign_sbc_cluster(adi: float, cv2: float) -> int:
    """Thesis numbering: 1 Smooth, 2 Intermittent, 3 Erratic, 4 Lumpy."""
    if pd.isna(adi) or pd.isna(cv2):
        return 4
    if adi < ADI_TH and cv2 < CV2_TH:
        return 1
    if adi >= ADI_TH and cv2 < CV2_TH:
        return 2
    if adi < ADI_TH and cv2 >= CV2_TH:
        return 3
    return 4


def compute_sbc_from_series(qty: pd.Series) -> dict:
    """Compute ADI, CV2, cluster from a single demand series (zeros included)."""
    qty = qty.astype(float)
    n = len(qty)
    n_pos = int((qty > 0).sum())
    adi = n / n_pos if n_pos > 0 else np.nan
    pos = qty[qty > 0]
    if len(pos) > 0 and pos.mean() > 0:
        cv2 = float((pos.std(ddof=1) / pos.mean()) ** 2) if len(pos) > 1 else 0.0
    else:
        cv2 = np.nan
    cluster = assign_sbc_cluster(adi, cv2)
    return {
        "ADI": adi,
        "CV2": cv2,
        "SBC_CLUSTER": cluster,
        "SBC_CLUSTER_LABEL": CLUSTER_LABELS[cluster],
        "n_periods": n,
        "n_positive": n_pos,
    }


def compute_sbc_table(df: pd.DataFrame, group_cols: list[str], qty_col: str = "sales") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = compute_sbc_from_series(g[qty_col])
        rows.append({**dict(zip(group_cols, keys)), **stats})
    return pd.DataFrame(rows)
