"""Generate notebook 14 — Optuna XGBoost tuning by type."""
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
        "# 14 Optuna 하이퍼파라미터 튜닝 (XGBoost)\n\n"
        "13장에서 **변수 중요도**를 본 XGBoost를, type별로 **Optuna**로 튜닝해 "
        "검증 구간 **MAPE**를 개선할 수 있는지 확인합니다.\n\n"
        "## 왜 하는가\n\n"
        "- 08·10장은 **고정 하이퍼파라미터** XGBoost — type마다 최적 구조가 다를 수 있음\n"
        "- **Optuna(TPE)** 로 탐색 공간을 자동 탐색 → 수작업 grid 대비 효율적\n"
        "- 튜닝 전·후 MAPE 비교로 **type별 개선 여지** 정량화\n\n"
        "| 항목 | 내용 |\n"
        "|------|------|\n"
        "| 모델 | XGBoost 패널 (type 내 family 합침) |\n"
        "| 목적함수 | 검증 3주 **MAPE** 최소화 |\n"
        "| 탐색 | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha/lambda` |\n"
        "| baseline | 08·13장과 동일 고정 파라미터 |\n\n"
        "**선행:** 02, 03, 13 | **산출:** `xgboost_optuna_tuning_results.csv`"
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
from utils.splits import VAL_WEEKS, TRAIN_WEEK_MAX
from utils.device import device_label
from utils.hyperparameter_tuning import (
    DEFAULT_XGB_PARAMS, panel_val_mape, tune_xgb_all_types,
)

N_TRIALS = 30  # type당 Optuna trial 수 (늘리면 정확도↑, 시간↑)

print('Torch device:', device_label())
feat_df = pd.read_parquet(DATA_PROCESSED / 'df_weekly_features.parquet')
types = sorted(feat_df['type'].unique())
print('types:', types)
print('학습 <=', TRAIN_WEEK_MAX, '| 검증:', VAL_WEEKS)
print('baseline params:', DEFAULT_XGB_PARAMS)
print('n_trials per type:', N_TRIALS)
"""
    ),
    md("### ① baseline MAPE (튜닝 전)"),
    code(
        """baseline_rows = []
for t in types:
    m = panel_val_mape(feat_df, t, None)
    baseline_rows.append({'type': t, 'baseline_mape': round(m, 4)})
baseline_df = pd.DataFrame(baseline_rows)
print(baseline_df.to_string(index=False))
"""
    ),
    md(
        "### ② Optuna 튜닝 — type별 validation MAPE 최소화\n\n"
        "type마다 독립 study. 패널 전체를 한 번 학습하고 검증 3주 예측 MAPE를 목적함수로 사용합니다."
    ),
    code(
        """results = tune_xgb_all_types(feat_df, types, n_trials=N_TRIALS)
display(results)
"""
    ),
    md("### ③ baseline vs tuned 비교"),
    code(
        """fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(types))
w = 0.35
ax.bar(x - w/2, results['baseline_mape'], w, label='Baseline', color='lightgray')
ax.bar(x + w/2, results['tuned_mape'], w, label='Tuned (Optuna)', color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels(results['type'])
ax.set_ylabel('Validation MAPE (%)')
ax.set_title('XGBoost MAPE: Baseline vs Optuna-tuned by Type')
ax.legend()
for i, row in results.iterrows():
    imp = row['improvement_pct']
    if pd.notna(imp):
        ax.annotate(f'{imp:+.1f}%', (i, row['tuned_mape']), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

print('\\n=== 개선 요약 ===')
print(results[['type', 'baseline_mape', 'tuned_mape', 'improvement_pct']].to_string(index=False))
"""
    ),
    md("### ④ type별 Best 하이퍼파라미터"),
    code(
        """param_cols = [c for c in results.columns if c.startswith('best_')]
best_params = results[['type'] + param_cols].copy()
best_params.columns = ['type'] + [c.replace('best_', '') for c in param_cols]
display(best_params)
"""
    ),
    md(
        "### ⑤ 해석 가이드\n\n"
        "- **improvement_pct > 0** → 해당 type에서 튜닝이 검증 MAPE를 개선\n"
        "- **개선이 작거나 음수** → 기본 파라미터가 이미 충분히 좋거나, 3주 검증이 짧아 과적합 가능\n"
        "- `learning_rate`↓ + `n_estimators`↑ 조합이 자주 나오면 → **세밀한 부스팅** 선호\n"
        "- type별 best params가 크게 다르면 → **매장 유형별 맞춤 모델** 운영 근거\n\n"
        "> 본 장은 **단일 validation window** 기준입니다. 운영 전에는 rolling CV 등으로 재검증하세요."
    ),
    code(
        """out = DATA_PROCESSED
results.to_csv(out / 'xgboost_optuna_tuning_results.csv', index=False)
param_cols = [c for c in results.columns if c.startswith('best_')]
best_only = results[['type'] + param_cols].rename(
    columns={c: c.replace('best_', '') for c in param_cols}
)
best_only.to_csv(out / 'xgboost_optuna_best_params_by_type.csv', index=False)
print('저장:', out / 'xgboost_optuna_tuning_results.csv')
"""
    ),
]

(ROOT / "14_Optuna_튜닝.ipynb").write_text(
    json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8"
)
print("wrote 14_Optuna_튜닝.ipynb")
