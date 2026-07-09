# AI와 소매 수요 예측 — 실습 저장소

Ecuador Favorita **Store Sales** 데이터로 소매 수요 예측의 전 과정(패턴 분석 → 분류·클러스터링 → 다종 알고리즘 벤치마크 → 임베딩 하이브리드 → scheme 비교)을 재현하는 Jupyter Notebook 실습 코드입니다.

📘 관련 도서: **AI와 소매 수요 예측** (김정혁, 커뮤니케이션북스) · 학위논문 「소매 수요 예측을 위한 클러스터링 기반 하이브리드 예측 방법론」의 축소 교육용 재현

---

## 왜 이 분석이 의미 있는가

소매 수요 예측의 목적은 **재고·발주·인력·프로모션** 결정에 쓸 **미래 판매량**을 정확히 추정하는 것입니다. 그러나 매장(type)·상품(family)마다 수요 패턴이 다릅니다.

- **연속·안정(Smooth)** vs **간헐·럼피(Intermittent/Lumpy)**
- **고변동** vs **저변동**
- **규칙 기반 분류(SBC)** 로 나눌 것인가, **ML 임베딩·클러스터링**으로 세분화할 것인가

본 저장소는 다음에 답합니다.

1. 제품 수요의 **변동 구조**(System-Level CV)는 type별로 어떻게 다른가? → **고변동/저변동 2-type 선정**
2. **10종 예측 알고리즘** 중 조건(type×cluster)마다 무엇이 유리한가?
3. **시계열 임베딩 6종**을 결합하면 예측이 개선되는가?
4. **SBC(rule-base) vs ML 클러스터링** 중 어느 scheme이 type별로 나은가?

---

## 데이터·실험 단위 (2-type로 축소)

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
| 원 분석 단위 | store type 5종 × family 33종 = **165 주간 시계열** (242주) |
| **실험 대상** | **02장에서 System-Level CV 기준 자동 선정한 2-type** — 고변동 **E**(CV 0.429) · 저변동 **C**(CV 0.259) → **66 시계열** |
| 논문 대응 | Center B(고변동)↔E, Center A(저변동)↔C |
| **분할 (3-way)** | **train ≤201707 (216주)** / **val 201708–201720 (13주, Optuna 튜닝)** / **test 201721–201733 (13주, 최종 평가)** |
| 예측 지평 | **13주** (lookback 26) |

> 논문·다이소 실증은 **12,661 SKU·2센터**. 본 실습은 **66 시계열**의 교육용 축소판이므로, 표본이 작고 변동 스펙트럼이 좁아 방법론 간 성능 차이가 논문만큼 크지 않을 수 있습니다.

데이터는 라이선스상 저장소에 포함되지 않습니다. Kaggle에서 받아 `data/raw/`에 넣고 `02`장부터 실행하세요.

---

## 핵심 설계 — 결과가 합리적으로 나오도록 한 수정

초기 구현은 몇 가지 문제로 비정상적 결과(예: iTransformer가 비상식적으로 낮은 성능, 롱테일 과대예측)를 냈습니다. 논문 방법론에 맞춰 아래를 수정했습니다.

| 수정 | 내용 | 효과 |
|------|------|------|
| **DL 정규화** | iTransformer/Autoformer 입력을 열별 **Min-Max 정규화**(논문 §4.2). 미정규화 시 대규모 판매량(GROCERY I ~26만)이 트랜스포머 학습을 붕괴 | iTransformer가 **중위권 경쟁력 회복**(median MAPE 33.8) |
| **패널 정규화** | RF/XGBoost 패널을 **family별 Min-Max**로 정규화 학습·역변환. 규모 1~265,000 이질 family를 한 패널에 학습 시 작은 family 과대예측 방지 | LAG 롱테일 MAE 390→7 등 정상화 |
| **검증 기반 튜닝** | **Optuna(TPE)** 로 XGBoost·iTransformer를 **val 13주 RMSE** 최소화 튜닝(논문 §4.4) | 과적합 방지·조건별 최적화 |
| **누수 제거** | lag/rolling을 예측값으로 갱신하는 **재귀 다단계 예측**으로 13주 예측 시 test 실측 누수 제거 | 공정한 다단계 평가 |
| **WMAPE 제품수 가중** | scheme 비교를 **Σ n_k·MAPE_k / Σ n_k**(논문 §4.5 Eq.50)로 통일 | 논문식 scheme 비교 |

---

## 사용 알고리즘

### 예측 (Phase 1, 10종 — 07·08·09장)

| 구분 | 알고리즘 |
|------|----------|
| 통계 (07) | ARIMA, Prophet, SBA, TSB |
| 머신러닝 (08) | Random Forest, XGBoost (family별 Min-Max + Optuna) |
| 딥러닝 (09) | LSTM, Autoformer, N-HiTS, iTransformer (공식 구현) |

### 딥러닝 (09장) — 공식 모델 코드

| 모델 | 연결 | 비고 |
|------|------|------|
| Autoformer, iTransformer | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (`vendor/` submodule) | Min-Max 정규화 + iTransformer는 Optuna 튜닝 |
| N-HiTS, LSTM | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) | 조건 내 다변량 패널 일괄 학습 |

### 시계열 임베딩 (06·10장) — 공식 repo

GAF-CNN([eliotwalt/gaf-cnn](https://github.com/eliotwalt/gaf-cnn)), TS2Vec([zhihanyue/ts2vec](https://github.com/zhihanyue/ts2vec)), PatchTST([PatchTST/PatchTST](https://github.com/PatchTST/PatchTST)) = vendor submodule / PCA, FastDTW, AE = 경량 baseline

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

### 클러스터링 (06장)

6 임베딩 × 4 방법(KMeans, HAC, GMM, DBSCAN) 그리드 → **균형 인지 선정**(degenerate 해 배격). 축소 데이터라 **K=2** 상한.
→ 채택: **AE + KMeans, [62, 4]** (cluster1 = GROCERY I·BEVERAGES·CLEANING 등 **메가셀러**, cluster2 = **롱테일**)

### 분류 (05장)

SBC (ADI 1.32 · CV² 0.49 rule-base 4클러스터 — Smooth/Intermittent/Erratic/Lumpy)

---

## 평가 지표

| 지표 | 용도 |
|------|------|
| MAE, RMSE, **MAPE**, MASE | 07–10장 알고리즘·임베딩 성능 (조건별 Best = MAPE 최소) |
| **WMAPE** = Σ n_k·MAPE_k / Σ n_k | 11장 type별 SBC vs ML scheme 비교 (**제품수 가중**, 논문 §4.5) |
| Silhouette, Davies-Bouldin, min_cluster_size | 06장 클러스터링 품질·균형 |
| CV, RIDR | 12장 수요 변동성·밴드 분석 |

---

## 노트북 흐름

```
02 전처리·2-type선정 → 03 피처 → 05 SBC → 06 ML클러스터링
   → 07 통계 → 08 ML → 09 DL → 10 임베딩 → 11 하이브리드 → 12 RIDR
   → 13 변수중요도 → 14 Optuna
```

| 장 | 하는 일 | 이유 |
|----|---------|------|
| 02 | 일→주 집계 + **type별 System-Level CV → 고변동 E·저변동 C 자동 선정** | 실험 대상 2-type 결정 |
| 03 | lag·rolling 등 피처 (2-type) | 패널 예측용 |
| 05 | SBC 4분류 라벨 | **rule-base 클러스터** 축 |
| 06 | 임베딩×클러스터링 그리드 → **AE+KMeans K=2** | **ML 클러스터** 축 |
| 07–09 | **12조건**(SBC 8 + ML 4) × 10모델, 조건별 MAPE Best | Phase1 |
| 10 | **LSTM 고정 × 6임베딩** (72조합, 임베딩=static exog) | 임베딩 비교, 11장 입력 |
| 11 | type별 **제품수 가중 WMAPE** SBC vs ML | scheme 우열 |
| 12 | CV·RIDR·수요 밴드 | scheme 차이의 변동성 근거 |
| 13 | type별 XGBoost 변수 중요도 | 피처 기여 (lag_1 지배) |
| 14 | type별 XGBoost **Optuna** 튜닝 | val 13주 MAPE 개선 |

---

## 주요 결과 (2-type 실습)

- **Phase1 10모델 median MAPE (31~40로 촘촘):** SBA 31.4 · N-HiTS 32.6 · ARIMA 32.9 · **iTransformer 33.8** · LSTM 34.0 · XGBoost 34.6 · TSB 34.6 · Prophet 36.5 · RF 36.9 · Autoformer 39.7. 정규화·튜닝 후 **iTransformer 정상화**. **mean MAPE는 LSTM 45.1로 최저**(robust) — ARIMA와 사실상 공동 선두.
- **Phase2 base = LSTM + 임베딩:** ARIMA가 Phase1 상위지만 **단변량이라 임베딩 결합 불가** → 임베딩으로 개선 가능한 실질 best인 **LSTM**을 base로 고정(임베딩=static exog, 논문 iTransformer+임베딩 계열). 임베딩 6종은 근소차(median ~34.4), 조건별 Best는 FastDTW 5개.
- **11장 scheme (제품수 가중 WMAPE) — 논문 가설과 방향 일치 ✅:** **C(저변동)→ML**(38.21 vs 38.72), **E(고변동)→SBC**(46.64 vs 46.87). 논문(저변동 센터 A→ML, 고변동 센터 B→SBC)의 변동성↔scheme 방향과 일치. 단 격차 <1 WMAPE로 유의성은 제한적.
  - (참고) base가 XGBoost일 땐 롱테일 폭주로 둘 다 SBC 우세였으나, **robust한 LSTM base에서 ML이 경쟁력을 회복**하며 논문 방향이 드러남 → scheme 우열은 base robust성에 민감.
- **13장 변수 중요도:** **lag_1 지배(71%)** → 논문 LAG1 top importance와 일치.
- **14장 Optuna:** C 26.5→21.4(19%↓), E 59.8→41.8(30%↓) — 튜닝의 검증 개선 확인.

---

## 저장소 구조

```
ai-retail-demandforecasting/
├── vendor/            # Time-Series-Library, ts2vec, PatchTST, gaf-cnn (submodule)
├── code/
│   ├── 02~14_*.ipynb  # 실습 노트북
│   ├── execute_notebook.py
│   └── utils/         # 공통 로직 (아래)
├── data/{raw,processed}/
└── docs/
```

### `code/utils/` 핵심 모듈

| 파일 | 역할 |
|------|------|
| `config.py` | 선정된 2-type 저장/로드(`selected_types.json`), 다운스트림 필터 |
| `stats_summary.py` | System/SKU-Level 변동성, **고/저변동 type 선정**, Mann-Whitney 검정 |
| `sbc.py` | ADI·CV² SBC 4분류 (1.32/0.49) |
| `clustering_experiments.py` | 임베딩×클러스터링 그리드, **균형 인지 선정**, outlier 재클러스터링 |
| `embeddings.py` | 6종 임베딩 레지스트리 (공식 vendor 어댑터) |
| `splits.py` | **3-way 분할**(train/val/test) + 클러스터링 전용 구간 |
| `forecasting.py` | ARIMA·Prophet·**정규화 패널 재귀예측**(`build_scaled_panel_training`, `scaled_recursive_forecast`) |
| `tslib_adapter.py` | thuml Autoformer·iTransformer (**Min-Max 정규화**) |
| `neuralforecast_adapter.py` | Nixtla NHITS·LSTM (Phase2는 **임베딩=static exog** 결합) |
| `tuning.py` | 조건 패널 **Optuna(TPE)** — XGBoost·iTransformer val RMSE |
| `phase_experiments.py` | **Phase1/2 엔진** (3-way, 튜닝, 재귀, 정규화) |
| `phase_analysis.py` | 11장 **제품수 가중 WMAPE**(`thesis_wmape_by_type`) |
| `metrics.py` | MAE/RMSE/MAPE/MASE/WMAPE |
| `feature_importance.py` | 13장 XGBoost 변수 중요도 |
| `hyperparameter_tuning.py` | 14장 type별 Optuna |

---

## 실행 방법

```bash
pip install -r requirements.txt
git submodule update --init --recursive
cd code
python execute_notebook.py "02_데이터셋_로드_전처리.ipynb"
# 02 → 03 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 순
```

Python 3.10+, GPU 권장(DL·임베딩 가속). 자세한 차이·해석: [`docs/실험_프레임워크.md`](docs/실험_프레임워크.md)

---

## 해석 시 유의사항

1. **논문 SOTA ≠ 축소 실습 Best** — 논문 iTransformer 강세는 12,661 SKU·풍부한 ML 군집 맥락. 66 시계열에서는 단순 모델도 경쟁적이며 iTransformer는 중위권.
2. **SBC vs ML 우열은 base 모델 robust성·데이터 규모에 민감** — robust한 LSTM base에서는 저변동→ML·고변동→SBC로 논문 방향과 일치(격차는 작음). XGBoost base처럼 롱테일에 취약한 base에선 결과가 뒤집힐 수 있음.
3. **MAPE는 간헐·near-zero 수요에서 불안정** — 롱테일 family의 MAPE는 크게 튈 수 있어 median·WMAPE를 함께 봄.
4. **단일 데이터셋** — Ecuador 2-type 결과가 모든 소매에 일반화되지는 않음.

---

## 라이선스

코드는 **연구·교육 목적**으로만 사용 가능합니다. 상업적 이용은 저자 사전 허가가 필요합니다.
