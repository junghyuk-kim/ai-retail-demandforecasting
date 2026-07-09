"""Train/validation/test split helpers for weekly panels.

3-way 시계열 분할 (논문 §4.4):
- Train : yearweek <= TRAIN_WEEK_MAX (216주) — 모델 학습 + 클러스터링
- Val   : VAL_WEEKS (13주) — Optuna 하이퍼파라미터 튜닝 (RMSE 최소화)
- Test  : TEST_WEEKS (13주) — 최종 정확도 (학습·튜닝에서 미사용)

최종 예측 시에는 train+val(<=TRAINVAL_WEEK_MAX)로 재학습 후 test를 예측한다.
"""
from __future__ import annotations

import pandas as pd

HORIZON = 13  # 예측 지평 = test/val 길이 (논문: 1개 분기)

# 전체 242주(201301~201733) 기준 마지막 26주를 val/test로 분리
TRAIN_WEEK_MAX = 201707
VAL_WEEKS = [
    201708, 201709, 201710, 201711, 201712, 201713, 201714,
    201715, 201716, 201717, 201718, 201719, 201720,
]
TEST_WEEKS = [
    201721, 201722, 201723, 201724, 201725, 201726, 201727,
    201728, 201729, 201730, 201731, 201732, 201733,
]
TRAINVAL_WEEK_MAX = 201720  # 최종 test 예측용 재학습 상한 (train+val)

# 클러스터링(06장) 전용 학습 구간 상한 — 예측 split과 분리.
# 신경망 임베딩은 확률적이라 재실행 시 클러스터가 바뀔 수 있어, 최초 실행값(201730)으로 고정.
CLUSTER_TRAIN_WEEK_MAX = 201730


def compute_split_weeks(df: pd.DataFrame, horizon: int = HORIZON, week_col: str = "yearweek") -> dict:
    """데이터의 실제 주차에서 3-way 분할 경계를 계산 (상수 대신 동적 사용 시)."""
    wk = sorted(int(x) for x in df[week_col].unique())
    test = wk[-horizon:]
    val = wk[-2 * horizon : -horizon]
    return {
        "train_week_max": wk[-2 * horizon - 1],
        "val_weeks": val,
        "test_weeks": test,
        "trainval_week_max": val[-1],
        "n_train": len(wk[: -2 * horizon]),
    }


def add_split_flag(df: pd.DataFrame, week_col: str = "yearweek") -> pd.DataFrame:
    out = df.copy()

    def _flag(w: int) -> str:
        if w in TEST_WEEKS:
            return "test"
        if w in VAL_WEEKS:
            return "val"
        return "train"

    out["split"] = out[week_col].apply(_flag)
    return out


def train_val_test_masks(df: pd.DataFrame, week_col: str = "yearweek"):
    """(train, val, test) boolean 마스크."""
    train_m = df[week_col] <= TRAIN_WEEK_MAX
    val_m = df[week_col].isin(VAL_WEEKS)
    test_m = df[week_col].isin(TEST_WEEKS)
    return train_m, val_m, test_m


def train_val_mask(df: pd.DataFrame, week_col: str = "yearweek") -> tuple[pd.Series, pd.Series]:
    """하위호환: (train, test) — 여기서 두 번째는 최종 평가 구간(TEST_WEEKS)."""
    train_m = df[week_col] <= TRAINVAL_WEEK_MAX
    test_m = df[week_col].isin(TEST_WEEKS)
    return train_m, test_m
