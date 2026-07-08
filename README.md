# AI와 소매 수요 예측 — 실습 저장소

Ecuador Favorita **Store Sales** 데이터를 활용해, 소매 수요 예측의 전 과정(패턴 분석 → 분류·클러스터링 → 다종 알고리즘 벤치마크 → 임베딩 하이브리드 → scheme 비교)을 재현하는 Jupyter Notebook 실습 코드입니다.

📘 관련 도서: **AI와 소매 수요 예측** (김정혁, 커뮤니케이션북스)

---

## 왜 이 분석이 의미 있는가

소매 수요 예측의 목적은 **재고·발주·인력·프로모션** 결정에 쓸 **미래 판매량**을 가능한 한 정확히 추정하는 것입니다. 그러나 매장(type)·상품(family)마다 수요 패턴이 다릅니다.

- **연속·안정(Smooth)** vs **간헐·럼피(Intermittent/Lumpy)**
- **고변동** vs **저변동**
- **규칙 기반 분류(SBC)** 로 나눌 것인가, **ML 임베딩·클러스터링**으로 세분화할 것인가

본 저장소는 다음 질문에 답합니다.

1. 제품 수요의 **변동 구조**(CV, RIDR)는 type·클러스터별로 어떻게 다른가?
2. **10종 예측 알고리즘** 중 조건(type×cluster)마다 무엇이 유리한가?
3. **시계열 임베딩 6종**을 결합하면 예측이 개선되는가?
4. 최종적으로 **SBC(rule-base) vs ML 클러스터링** 중 어느 scheme이 type별로 나은가?

> SBC 4분류(Smooth→ARIMA 등)는 **이론적 권고**이며, 본 실습에서는 **실측 지표로 Best를 선택**합니다.

**논문 vs 실습:** 학위논문에서는 iTransformer 등 DL이 강세였으나, 본 Ecuador 실습(165 주간 시계열·lookback 16·3주 검증·MAPE Best)에서는 **XGBoost가 10모델 중 가장 자주 선택**됩니다. lag·rolling 피처가 강하고(13장), DL은 논문 대비 variate 수·입력 길이가 작습니다. 자세한 비교는 [`docs/실험_프레임워크.md`](docs/실험_프레임워크.md) 참고.

---

## 데이터·실험 단위

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
| 분석 단위 | **type × family** 주간 시계열 **165개** |
| type (5) | D, B, C, E, A — 논문의 센터 A/B 등에 대응 |
| 학습 | `yearweek ≤ 201730` |
| 검증 | `201731–201733` (3주) |

데이터는 라이선스상 저장소에 포함되지 않습니다. Kaggle에서 받아 `data/`에 넣은 뒤 `02`장부터 실행하세요.

---

## 사용 알고리즘

### 예측 (Phase 1, 10종 — 07·08·09장)

| 구분 | 알고리즘 |
|------|----------|
| 통계 (07) | ARIMA, Prophet, SBA, TSB |
| 머신러닝 (08) | Random Forest, XGBoost |
| 딥러닝 (09) | LSTM, Autoformer, N-HiTS, iTransformer (아래 공식 구현) |

### 딥러닝 (09장) — 공식 모델 코드

| 모델 | 연결 방식 | 비고 |
|------|-----------|------|
| Autoformer, iTransformer | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (`vendor/` submodule) | 단독 [Autoformer](https://github.com/thuml/Autoformer)·[iTransformer](https://github.com/thuml/iTransformer) repo의 `run.py`가 아닌 **TSlib `Model` 클래스** |
| N-HiTS, LSTM | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) pip | 조건 내 다변량 패널 일괄 학습 |

- 학습: type×cluster 조건당 **family 다변량 패널** 1회 (`tslib_adapter.py`, `neuralforecast_adapter.py`)
- 설정: lookback **16**, 검증 **3주**, Phase1 Best = **MAPE**

```bash
git submodule update --init --recursive   # Time-Series-Library 받기
pip install -r requirements.txt
```

### 시계열 임베딩 (10장)

PCA, FastDTW, AutoEncoder, GAF-CNN, TS2Vec, PatchTST

### 클러스터링 (06장)

6 임베딩 × 4 방법 (KMeans, HAC, GMM, DBSCAN) = 24조합 → **TS2Vec + KMeans** 채택

### 분류 (05장)

SBC (ADI·CV² rule-base 4클러스터)

---

## 평가 지표

| 지표 | 용도 |
|------|------|
| **MAE, RMSE, MAPE, MASE** | 07–10장 알고리즘·임베딩 성능 (조건별 Best = **MAPE** 최소) |
| **WMAPE** (가중) | 11장만 — type별 SBC vs ML scheme 비교 (제품 검증판매량 가중) |
| Silhouette, Davies-Bouldin | 06장 클러스터링 품질 |
| CV, RIDR | 12장 수요 변동성·밴드 분석 |

---

## 노트북 흐름 — 각 장을 왜 하는가

```
02 전처리 → 03 피처 → 04 패턴개요 → 05 SBC → 06 ML클러스터링
    → 07 통계 → 08 ML → 09 DL → 10 임베딩 → 11 하이브리드 → 12 RIDR
    → 13 변수중요도 → 14 Optuna
```

| 장 | 파일 | 하는 일 | 이유 |
|----|------|---------|------|
| 02 | `02_데이터셋_로드_전처리.ipynb` | 일→주 집계, type×family 패널 | 예측·클러스터링의 공통 입력 |
| 03 | `03_특징_엔지니어링.ipynb` | lag·rolling 등 피처 | RF/XGBoost 패널 예측용 |
| 04 | `04_수요패턴_분석.ipynb` | 패턴 개요·기초 통계 | 12장 상세 분석의 입문 |
| 05 | `05_rule_base_분류.ipynb` | SBC 4분류 라벨 | **rule-base 클러스터** 축 생성 |
| 06 | `06_클러스터링_방법.ipynb` | 24조합 그리드 → ML_CLUSTER | **ML 클러스터** 축 생성 (TS2Vec+KMeans) |
| 07 | `07_통계_예측모델.ipynb` | 40조건×ARIMA/Prophet/SBA/TSB | Phase1 1/3 — 통계·간헐 수요 |
| 08 | `08_머신러닝_예측모델.ipynb` | 40조건×RF/XGBoost | Phase1 2/3 — 피처 기반 패널 ML |
| 09 | `09_딥러닝_예측모델.ipynb` | 40조건×DL 4종 + **10모델 통합 Best** | 공식 DL + Phase1 3/3 (조건별 MAPE Best) |
| 10 | `10_임베딩_기반_시계열분석.ipynb` | **XGBoost 고정** × 6임베딩, 40조건 | 임베딩 방법 비교(실험 수 축소), 11장 입력 |
| 11 | `11_하이브리드_수요예측.ipynb` | type별 **가중 WMAPE** SBC vs ML | **어느 clustering scheme이 유리한지** 최종 비교 |
| 12 | `12_제품_수요패턴_RIDR_분석.ipynb` | CV·RIDR·수요 밴드, 11장 연계 | scheme 차이의 **변동성 근거** (논문 3장 스타일) |
| 13 | `13_변수_중요도_분석.ipynb` | type별 **XGBoost 변수 중요도** | lag·외생변수 등 **어떤 피처가 예측에 기여하는지** |
| 14 | `14_Optuna_튜닝.ipynb` | type별 XGBoost **Optuna** 튜닝 | validation MAPE 기준 하이퍼파라미터 자동 탐색 |

### 실험 조건 (07–10)

- **40조건** = 5 type × 4 cluster × 2 scheme (**SBC** + **ML**)
- 시계열이 없는 조합은 제외 → 유효 **35조건** 정도
- 10장: Phase1 Best가 조건마다 달라 실험 폭발 → **XGBoost(다수 조건 Best)** 를 대표 base로 고정해 6임베딩 비교
- 09장 DL 재실행(공식 구현) 후에도 **10모델 통합 Best는 XGBoost가 가장 빈번** — 논문(iTransformer 강세)과 실습 조건이 다름

### 11장 핵심 결론 (예시)

- 가중 WMAPE: **SBC 3 : ML 2** (type A,B,E → SBC / C,D → ML)
- **CV·RIDR**과 대조 시 고변동 type(B,E)에서 SBC, 저변동(D)에서 ML 경향 — 논문 Center B/A 스토리와 **부분 정합** (항상 성립하지는 않음)

---

## 저장소 구조

```
ai-retail-demandforecasting/
├── vendor/
│   └── Time-Series-Library/   # git submodule (thuml 공식 Autoformer·iTransformer)
├── code/
│   ├── 02~14_*.ipynb          # 실습 노트북 (13 변수중요도, 14 Optuna)
│   ├── execute_notebook.py    # 노트북 일괄 실행·nbformat 수정
│   ├── build_phase_notebooks.py  # 07–11 노트북 템플릿 생성
│   ├── run_phase2_pipeline.py    # 10장 Phase2 CLI 실행(선택)
│   └── utils/                 # 공통 로직 (아래 표 참고)
├── data/
│   ├── raw/                   # Kaggle 원본 (사용자 준비)
│   └── processed/             # parquet/csv 산출물 (.gitignore)
└── docs/
    ├── 00_개발환경_셋팅.md
    └── 실험_프레임워크.md       # 논문 vs 실습 차이, DL 구현, 해석 가이드
```

---

## Python 모듈 (`code/utils/`) 요약

노트북이 반복하는 로직을 모듈로 분리했습니다.

### 경로·데이터

| 파일 | 역할 |
|------|------|
| `paths.py` | `DATA_PROCESSED`, `SBC_CLUSTER`, `ML_CLUSTER` 등 경로 상수 |
| `weekly.py` | 일별→주별 집계, `yearweek` 생성 |
| `splits.py` | 학습/검증 주차 (`TRAIN_WEEK_MAX`, `VAL_WEEKS`) |
| `experiment_data.py` | 주간 데이터 + 피처 + SBC/ML 라벨 일괄 로드 |

### SBC·패턴 분석

| 파일 | 역할 |
|------|------|
| `sbc.py` | ADI·CV² 계산, SBC 4클러스터 할당 (Smooth/Intermittent/Erratic/Lumpy) |
| `stats_summary.py` | type별 System-Level·SKU-Level 주간 통계 |
| `pattern_analysis.py` | 클러스터별 System/SKU 통계, **RIDR**·10–90% 밴드, type별 시각화 |

### 클러스터링 (06장)

| 파일 | 역할 |
|------|------|
| `embeddings.py` | 6종 임베딩 함수 + `EMBEDDERS` 레지스트리 (GPU 지원) |
| `clustering_experiments.py` | 6×4 그리드 실험, 실루엣·DB index, **joint rank**로 Best 조합 선정 |
| `metrics.py` | `clustering_quality()` — Silhouette, Davies-Bouldin |

### 예측 (07–10장)

| 파일 | 역할 |
|------|------|
| `forecasting.py` | ARIMA, Prophet, RF/XGBoost 패널 예측 |
| `intermittent.py` | SBA, TSB (간헐 수요) |
| `dl_models.py` | DL 조건 단위 학습 진입점 → `tslib_adapter` / `neuralforecast_adapter` |
| `tslib_adapter.py` | thuml Time-Series-Library **Autoformer·iTransformer** 학습·추론 |
| `neuralforecast_adapter.py` | Nixtla **NHITS·LSTM** 조건 패널 학습 |
| `device.py` | CUDA 자동 감지, DataLoader `pin_memory` |
| `metrics.py` | MAE, RMSE, MAPE, MASE, WMAPE, `forecast_metrics()` |
| `phase_experiments.py` | **Phase1/2 실험 엔진**: 40조건 루프, 4지표 저장, XGBoost 고정 Phase2, 전역 임베딩 캐시 |
| `phase_analysis.py` | 11장 **가중 WMAPE**, type별 SBC vs ML 비교, CV 연계표 |
| `feature_importance.py` | 13장 type별 **XGBoost feature importance** |
| `hyperparameter_tuning.py` | 14장 type별 **Optuna XGBoost 튜닝**, baseline vs tuned MAPE |

### `phase_experiments.py` 핵심 흐름

```
Phase1: run_phase1_all() → 조건(type×cluster)×모델 → 제품별 MAE/RMSE/MAPE/MASE
        merge_phase1_best() → 07+08+09 합쳐 10모델 중 MAPE Best

Phase2: build_global_embedding_cache() → 165시계열×6임베딩 1회 계산
        run_phase2_all(fixed_model='XGBoost') → XGBoost+임베딩 240조합
        summarize_phase2() → 조건별 Best 임베딩
```

### 보조 스크립트 (`code/`)

| 파일 | 역할 |
|------|------|
| `execute_notebook.py` | `python execute_notebook.py 노트북.ipynb [timeout]` — 실행 후 stream `name` 등 nbformat v5 수정 |
| `build_phase_notebooks.py` | 07–11 노트북 JSON 재생성 |
| `run_phase2_pipeline.py` | 10장 Phase2를 노트북 없이 CLI로 실행 (선택) |

---

## 실행 방법

### 1. 환경

```bash
pip install -r requirements.txt
```

Python 3.10+, Jupyter. GPU 있으면 DL·임베딩·XGBoost 가속. 자세한 설정: [`docs/00_개발환경_셋팅.md`](docs/00_개발환경_셋팅.md)

### 2. 권장 순서

1. `02` → `03` → `05` → `06` (데이터·라벨 준비)
2. `07` → `08` → `09` (Phase1, 09에서 10모델 Best 확정)
3. `10` (XGBoost×6임베딩)
4. `11` (SBC vs ML 가중 WMAPE)
5. `12` (RIDR·변동성 근거) — `04`는 선택적 개요
6. `13` (XGBoost 변수 중요도)
7. `14` (Optuna 하이퍼파라미터 튜닝)

### 3. 노트북 CLI 실행 (선택)

```bash
cd code
python execute_notebook.py "07_통계_예측모델.ipynb" 14400
```

`data/processed/`에 캐시가 있으면 해당 장은 로드만 하고 건너뜁니다.

---

## 해석 시 유의사항

1. **논문 ≠ 실습 Best** — 논문 iTransformer 강세와 Ecuador 실습(XGBoost 우세)은 **데이터·lookback·지표·variate 수** 차이로 설명 가능
2. **조건별 Best ≠ 전역 1위** — XGBoost가 가장 자주 이겨도 type×cluster마다 최적 모델은 다름
3. **SBC 라벨 ≠ 최적 알고리즘** — 분류는 해석·세분화 축, 선택은 실측 MAPE/WMAPE
4. **CV·RIDR ↔ scheme** — 고변동→SBC, 저변동→ML **경향**은 있으나 C type 등 예외 존재
5. **DL 공식 코드** — TSlib/neuralforecast **모델 클래스** 사용; 논문 벤치마크 `run.py` 파이프라인과 동일하지 않음
6. **단일 데이터셋** — Ecuador 5 type 결과가 모든 소매 데이터에 일반화되지는 않음

---

## 라이선스

코드는 **연구·교육 목적**으로만 사용 가능합니다. 상업적 이용은 저자 사전 허가가 필요합니다.
