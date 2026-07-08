"""Generate notebook 13 — XGBoost feature importance by type."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.6"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": s.splitlines(keepends=True),
    }


cells = [
    md(
        "# 13 변수 중요도 분석 (XGBoost)\n\n"
        "08·10장에서 핵심 예측 모델인 **XGBoost**를 type별로 학습하고, "
        "**어떤 피처가 판매량 예측에 기여하는지** 변수 중요도(feature importance)를 분석합니다.\n\n"
        "## 왜 하는가\n\n"
        "- 소매 수요예측에서 **lag·rolling·외생변수** 중 무엇이 type마다 다르게 작용하는지 파악\n"
        "- 10장 XGBoost+임베딩 하이브리드 해석의 **피처 축** 이해\n"
        "- 14장 Optuna 튜닝 전 **현재 고정 하이퍼파라미터** 모델의 변수 구조 파악\n\n"
        "| 항목 | 내용 |\n"
        "|------|------|\n"
        "| 모델 | XGBoost (08장과 동일 하이퍼파라미터) |\n"
        "| 단위 | **type 5개 각각** 독립 학습 |\n"
        "| 데이터 | `df_weekly_features.parquet` 학습 구간 |\n"
        "| 지표 | `feature_importances_` (gain 비율, % 정규화) |\n\n"
        "**선행:** 02, 03 (피처 생성) | **다음:** 14 Optuna 튜닝"
    ),
    md("### ⓪ 환경 설정"),
    code(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NOTEBOOK_DIR = Path.cwd()
REPO_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'code' else NOTEBOOK_DIR
sys.path.insert(0, str(REPO_ROOT / 'code'))

from utils.paths import DATA_PROCESSED
from utils.splits import TRAIN_WEEK_MAX
from utils.device import device_label
from utils.feature_importance import (
    feature_cols, importance_by_type, importance_pivot,
)

print('Torch device:', device_label())
feat_df = pd.read_parquet(DATA_PROCESSED / 'df_weekly_features.parquet')
types = sorted(feat_df['type'].unique())
cols = feature_cols(feat_df)
print('types:', types)
print('피처 수:', len(cols), '| 학습 <=', TRAIN_WEEK_MAX)
"""
    ),
    md(
        "### ① type별 XGBoost 학습 · 변수 중요도\n\n"
        "각 type 내 모든 family 시계열을 **패널**로 합쳐 XGBoost 1개를 학습합니다 (08장 `forecast_ml_panel`과 동일 구조)."
    ),
    code(
        """imp_long = importance_by_type(feat_df, types)
imp_pivot = importance_pivot(imp_long)

print('=== type별 Top 10 피처 ===')
for t in types:
    top = imp_long[imp_long['type'] == t].nlargest(10, 'importance_pct')
    print(f'\\n--- Type {t} ---')
    print(top.to_string(index=False))

display(imp_pivot.round(2))
"""
    ),
    md("### ② 시각화 — type별 Top 15 피처"),
    code(
        """TOP_K = 15
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, t in enumerate(types):
    ax = axes[i]
    sub = imp_long[imp_long['type'] == t].nlargest(TOP_K, 'importance_pct')
    ax.barh(sub['feature'][::-1], sub['importance_pct'][::-1], color='steelblue')
    ax.set_title(f'Type {t}')
    ax.set_xlabel('Importance (%)')

axes[-1].set_visible(False)
fig.suptitle('XGBoost Feature Importance by Type (Top 15)', y=1.02)
plt.tight_layout()
plt.show()
"""
    ),
    md("### ③ type 간 공통·차별 피처"),
    code(
        """# type 평균 중요도
mean_imp = imp_pivot.mean(axis=1).sort_values(ascending=False)
print('=== 5 type 평균 Top 15 ===')
print(mean_imp.head(15).round(2))

# type 간 중요도 표준편차 — type마다 기여도가 다른 피처
std_imp = imp_pivot.std(axis=1).sort_values(ascending=False)
hetero = pd.DataFrame({
    'mean_pct': mean_imp,
    'std_across_types': std_imp,
}).sort_values('std_across_types', ascending=False)
print('\\n=== type 간 중요도 편차 큰 피처 (차별적) ===')
print(hetero.head(10).round(2))
"""
    ),
    md(
        "### ④ 해석 가이드\n\n"
        "- **lag·roll_mean** 계열이 상위면 → 해당 type은 **과거 판매 수준·추세**가 예측의 핵심\n"
        "- **zero_ratio·ADI·CV²** 상위면 → **간헐 수요** 특성이 모델에 반영\n"
        "- **oil·holiday·transactions** 상위면 → **외생·매장 활동** 변수가 type별로 설명력 보유\n"
        "- type 간 `std_across_types`가 크면 → **매장 유형마다 다른 드라이버** (SBC/ML scheme 차이 해석에 참고)\n\n"
        "> 본 분석은 **튜닝 전 고정 하이퍼파라미터** 기준입니다. 14장 Optuna 후 중요도가 달라질 수 있습니다."
    ),
    code(
        """out = DATA_PROCESSED
imp_long.to_csv(out / 'xgboost_feature_importance_by_type.csv', index=False)
imp_pivot.round(4).to_csv(out / 'xgboost_feature_importance_pivot.csv')
print('저장:', out / 'xgboost_feature_importance_by_type.csv')
"""
    ),
]

(ROOT / "13_변수_중요도_분석.ipynb").write_text(
    json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8"
)
print("wrote 13_변수_중요도_분석.ipynb")
