"""Generate notebooks 07-11 for restructured experiment pipeline."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.6"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source.splitlines(keepends=True),
    }


SETUP = """import sys
from pathlib import Path
import pandas as pd

NOTEBOOK_DIR = Path.cwd()
REPO_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'code' else NOTEBOOK_DIR
sys.path.insert(0, str(REPO_ROOT / 'code'))

from utils.paths import DATA_PROCESSED
from utils.splits import VAL_WEEKS, TRAIN_WEEK_MAX
from utils.device import device_label
from utils.experiment_data import load_forecast_frames
from utils.phase_experiments import (
    STAT_MODELS, ML_MODELS, DL_MODELS, RANK_METRIC,
    run_phase1_all, summarize_phase1, merge_phase1_best,
    build_global_embedding_cache, run_phase2_all, summarize_phase2,
    pick_final_per_condition,
)

print('Torch device:', device_label())
df, feat_df = load_forecast_frames()
print('시계열:', df.groupby(['type', 'family']).ngroups)
print('학습 <=', TRAIN_WEEK_MAX, '| 검증:', VAL_WEEKS)
print('Best 선정 기준:', RANK_METRIC.upper())
"""


def phase1_run_block(models_var: str, cache_name: str, label: str) -> str:
    return f"""# Phase 1 — {label}: SBC 20조건 + ML 20조건
cache = DATA_PROCESSED / '{cache_name}'
if cache.exists():
    results = pd.read_parquet(cache)
    summary, best = summarize_phase1(results)
    print('캐시 로드 |', len(results), 'rows')
else:
    sbc = run_phase1_all(df, feat_df, 'SBC_CLUSTER', 'SBC', models={models_var})
    ml = run_phase1_all(df, feat_df, 'ML_CLUSTER', 'ML', models={models_var})
    results = pd.concat([sbc, ml], ignore_index=True)
    summary, best = summarize_phase1(results)
    results.to_parquet(cache, index=False)
    summary.to_csv(DATA_PROCESSED / '{cache_name.replace(".parquet", "_summary.csv")}', index=False)
    best.to_csv(DATA_PROCESSED / '{cache_name.replace(".parquet", "_best.csv")}', index=False)
    print('완료 |', len(results), 'rows')

display(best.sort_values(['cluster_scheme', 'type', 'cluster']))
print('\\n=== 알고리즘별 평균', RANK_METRIC.upper(), '===')
print(results.groupby(['cluster_scheme', 'model'])[RANK_METRIC].mean().unstack('cluster_scheme').round(2))
"""


# --- 07 Statistical ---
cells07 = [
    md(
        "# 07 통계 예측 모델 (Phase 1 — 1/3)\n\n"
        "**40조건** (SBC 20 + ML 20) × **ARIMA, Prophet, SBA, TSB**\n\n"
        "- 오차지표: MAE, RMSE, MAPE, MASE (조건별 Best는 **MAPE** 기준)\n"
        "- WMAPE는 11장 하이브리드 비교에서만 사용"
    ),
    md("### ⓪ 환경 설정"),
    code(SETUP),
    md("### ① 통계 모델 실험"),
    code(phase1_run_block("STAT_MODELS", "phase1_stat_results.parquet", "통계 4종")),
]

# --- 08 ML ---
cells08 = [
    md(
        "# 08 머신러닝 예측 모델 (Phase 1 — 2/3)\n\n"
        "**40조건** × **RF, XGBoost** | 지표: MAE, RMSE, MAPE, MASE"
    ),
    md("### ⓪ 환경 설정"),
    code(SETUP),
    md("### ① ML 모델 실험"),
    code(phase1_run_block("ML_MODELS", "phase1_ml_results.parquet", "ML 2종")),
]

# --- 09 DL + merge ---
cells09 = [
    md(
        "# 09 딥러닝 예측 모델 (Phase 1 — 3/3)\n\n"
        "**40조건** × **LSTM, Autoformer, N-HiTS, iTransformer**\n\n"
        "07·08 결과와 합쳐 **10모델 중 조건별 Best** 선정"
    ),
    md("### ⓪ 환경 설정"),
    code(SETUP),
    md("### ① DL 모델 실험"),
    code(phase1_run_block("DL_MODELS", "phase1_dl_results.parquet", "DL 4종")),
    md("### ② Phase 1 통합 Best (10모델)"),
    code(
        """stat = pd.read_parquet(DATA_PROCESSED / 'phase1_stat_results.parquet')
ml = pd.read_parquet(DATA_PROCESSED / 'phase1_ml_results.parquet')
dl = pd.read_parquet(DATA_PROCESSED / 'phase1_dl_results.parquet')

phase1_all = pd.concat([stat, ml, dl], ignore_index=True)
phase1_summary, phase1_best = merge_phase1_best(stat, ml, dl)
phase1_all.to_parquet(DATA_PROCESSED / 'phase1_all_results.parquet', index=False)
phase1_summary.to_csv(DATA_PROCESSED / 'phase1_summary.csv', index=False)
phase1_best.to_csv(DATA_PROCESSED / 'phase1_best_per_condition.csv', index=False)
print('Phase1 통합 Best |', len(phase1_best), '조건')
display(phase1_best.sort_values(['cluster_scheme', 'type', 'cluster']))
"""
    ),
]

# --- 10 Embedding Phase2 ---
cells10 = [
    md(
        "# 10 임베딩 기반 시계열 분석 (Phase 2)\n\n"
        "Phase1 Best 알고리즘 × **6 임베딩** (PCA, FastDTW, AE, GAF-CNN, TS2Vec, PatchTST)\n\n"
        "- 오차지표: MAE, RMSE, MAPE, MASE → 조건별 최적 임베딩 조합 선정"
    ),
    md("### ⓪ 환경 설정"),
    code(SETUP + "\nphase1_best = pd.read_csv(DATA_PROCESSED / 'phase1_best_per_condition.csv')\n"),
    md("### ① Phase 2 하이브리드 실험"),
    code(
        """p2_cache = DATA_PROCESSED / 'phase2_results.parquet'
if p2_cache.exists():
    phase2 = pd.read_parquet(p2_cache)
    phase2_summary, phase2_best = summarize_phase2(phase2)
    print('Phase2 캐시 로드 |', len(phase2))
else:
    emb_cache = build_global_embedding_cache(df)
    p2_sbc = run_phase2_all(df, feat_df, 'SBC_CLUSTER', 'SBC', phase1_best, emb_cache=emb_cache)
    p2_ml = run_phase2_all(df, feat_df, 'ML_CLUSTER', 'ML', phase1_best, emb_cache=emb_cache)
    phase2 = pd.concat([p2_sbc, p2_ml], ignore_index=True)
    phase2_summary, phase2_best = summarize_phase2(phase2)
    phase2.to_parquet(p2_cache, index=False)
    phase2_summary.to_csv(DATA_PROCESSED / 'phase2_summary.csv', index=False)
    phase2_best.to_csv(DATA_PROCESSED / 'phase2_best_per_condition.csv', index=False)

display(phase2_best.sort_values(['cluster_scheme', 'type', 'cluster']))
print('\\n=== Best 임베딩 빈도 ===')
print(phase2_best['embedding'].value_counts())

final_best = pick_final_per_condition(phase1_best, phase2_best)
final_best.to_csv(DATA_PROCESSED / 'final_best_per_condition.csv', index=False)
print('Phase1 vs Phase2 최종 | Phase2 승률:', round((final_best.winner=='Phase2').mean(), 3))
"""
    ),
]

# --- 11 Hybrid WMAPE ---
cells11 = [
    md(
        "# 11 하이브리드 수요예측\n\n"
        "**type별 SBC(rule-base) vs ML 클러스터링** — 제품수(검증 판매량) 반영 **가중 WMAPE** 비교\n\n"
        "> 논문: 변동성 큰 **B센터**에서는 SBC가, 저변동 **A센터**에서는 ML이 유리했음.  \n"
        "> 본 데이터는 type별 변동성 차이가 상대적으로 작아 **결과가 항상 같지 않으며**, "
        "센터·데이터 특성에 따라 우세 scheme이 달라질 수 있음."
    ),
    md("### ⓪ 환경 설정"),
    code(
        SETUP
        + """
phase1_all = pd.read_parquet(DATA_PROCESSED / 'phase1_all_results.parquet')
phase2 = pd.read_parquet(DATA_PROCESSED / 'phase2_results.parquet')
phase1_best = pd.read_csv(DATA_PROCESSED / 'phase1_best_per_condition.csv')
phase2_best = pd.read_csv(DATA_PROCESSED / 'phase2_best_per_condition.csv')
final_best = pd.read_csv(DATA_PROCESSED / 'final_best_per_condition.csv')
"""
    ),
    md("### ① 가중 WMAPE — type별 SBC vs ML"),
    code(
        """import numpy as np
from utils.phase_analysis import (
    validation_weights, build_family_final_results,
    compare_schemes_by_type, summarize_wmape,
)

val_weights = validation_weights(df)
family_final = build_family_final_results(phase1_all, phase2, final_best, val_weights)

type_compare = compare_schemes_by_type(family_final)
display(type_compare)
print('가중 WMAPE 우세:', type_compare['better_scheme_weighted'].value_counts().to_dict())

cv = df[df['yearweek'] <= TRAIN_WEEK_MAX].groupby('type')['sales'].agg(['mean', 'std'])
cv['cv'] = (cv['std'] / cv['mean'].replace(0, np.nan)).round(3)
print('\\n=== type별 학습구간 판매 변동계수(CV) ===')
print(cv)
"""
    ),
    md(
        "### 해석\n\n"
        "- **가중 WMAPE** = Σ(WMAPE_f × 검증판매량_f) / Σ(검증판매량_f)\n"
        "- 논문과 달리 본 실험에서는 type 간 변동성 격차가 크지 않을 수 있음 → scheme 우세가 센터마다 다르게 나타남\n"
        "- `final_best_per_condition.csv` → 12장 RIDR 분석 이후 활용"
    ),
]

OUT = {
    "07_통계_예측모델.ipynb": cells07,
    "08_머신러닝_예측모델.ipynb": cells08,
    "09_딥러닝_예측모델.ipynb": cells09,
    "10_임베딩_기반_시계열분석.ipynb": cells10,
    "11_하이브리드_수요예측.ipynb": cells11,
}

if __name__ == "__main__":
    for name, cells in OUT.items():
        path = ROOT / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path)
