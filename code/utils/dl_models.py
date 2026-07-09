"""Deep learning forecasters — official model code via adapters.

- Autoformer, iTransformer: thuml/Time-Series-Library (vendor submodule) → tslib_adapter
- N-HiTS, LSTM: Nixtla/neuralforecast → neuralforecast_adapter

조건(type×cluster) 단위 다변량 패널 학습. 학습 진입은 phase_experiments에서
train_tslib_condition / train_nf_condition(_static)을 직접 호출한다.
"""
from __future__ import annotations

import pandas as pd

TSLIB_MODELS = {"Autoformer", "iTransformer"}
NF_MODELS = {"N-HiTS", "LSTM"}
DL_MODEL_NAMES = TSLIB_MODELS | NF_MODELS


def condition_wide(df: pd.DataFrame) -> pd.DataFrame:
    """type×cluster 조건 내 family×week 판매 wide matrix."""
    wide = df.pivot_table(index="yearweek", columns="family", values="sales", aggfunc="sum")
    return wide.sort_index().fillna(0)
