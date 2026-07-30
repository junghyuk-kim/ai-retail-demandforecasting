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

## 핵심 설계 원칙 (논문 방법론 반영)

소매 수요 예측에서 결과의 타당성을 좌우하는 아래 설계를 논문 방법론(§4.2·4.4·4.5)에 맞춰 반영했습니다.

| 원칙 | 내용 | 근거 |
|------|------|------|
| **DL 정규화** | iTransformer/Autoformer 입력을 열별 **Min-Max 정규화**(논문 §4.2) | 대규모 판매량(GROCERY I ~26만) 없이 트랜스포머 안정 학습(iTransformer median MAPE 51.6) |
| **패널 정규화** | RF/XGBoost 패널을 **family별 Min-Max**로 정규화 학습·역변환 | 규모 1~265,000의 이질 family를 동일 스케일에서 학습(롱테일 예측 안정) |
| **검증 기반 튜닝** | **Optuna(TPE)** — Phase1 XGBoost·iTransformer(val RMSE) + 14장 대표모델 **LSTM+임베딩**(val MAPE) | 과적합 방지·조건별 최적화(논문 §4.4) |
| **누수 제거** | lag/rolling을 예측값으로 갱신하는 **재귀 다단계 예측** | 13주 예측 시 test 실측 참조 누수 제거 → 공정한 다단계 평가 |
| **WMAPE 제품수 가중** | scheme 비교를 **Σ n_k·MAPE_k / Σ n_k**로 통일 | 논문 §4.5 Eq.50과 동일한 scheme 비교 |

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
| 04 | System/SKU-Level·SBC 미리보기 (2-type 개요) | 선택적 EDA |
| 05 | SBC 4분류 라벨 | **rule-base 클러스터** 축 |
| 06 | 임베딩×클러스터링 그리드 → **TS2Vec+KMeans K=2** | **ML 클러스터** 축 |
| 07–09 | **12조건**(SBC 8 + ML 4) × 10모델, 조건별 MAPE Best | Phase1 |
| 10 | **LSTM 고정 × 6임베딩** (72조합, 임베딩=static exog) | 임베딩 비교, 11장 입력 |
| 11 | type별 **제품수 가중 WMAPE** SBC vs ML | scheme 우열 |
| 12 | CV·RIDR·수요 밴드 | scheme 차이의 변동성 근거 |
| 13 | type별 XGBoost 변수 중요도 | 피처 기여 (lag_1 지배) |
| 14 | type별 **LSTM+임베딩 Optuna** 튜닝 (대표모델) | val 13주 MAPE 개선 |

---

## 주요 결과 (2-type 실습)

- **Phase1 10모델 median MAPE (12조건):** **ARIMA 41.7** · **LSTM 42.6** · SBA 45.6 · N-HiTS 46.4 · XGBoost 48.2 · RF 48.4 · TSB 48.5 · **iTransformer 51.6** · Prophet 56.0 · **Autoformer 73.1**. 조건별 승자는 **통계 7**(ARIMA 5·Prophet 2) · **딥러닝 3**(LSTM 2·N-HiTS 1) · **머신러닝 2**(XGBoost 1·RF 1).
- **밀린 것은 딥러닝 전체가 아니라 트랜스포머 계열:** LSTM은 2위, N-HiTS는 4위로 XGBoost·RF보다 앞섭니다. 하위권은 파라미터가 많은 iTransformer(8위)·Autoformer(10위) 둘뿐입니다. 66계열이라는 표본으로는 어텐션 기반 대형 모델이 학습되지 않습니다.
- **Phase2 base = LSTM + 임베딩 — ARIMA 단일보다 우수:** median MAPE **ARIMA 41.71 → LSTM 단일 42.58 → LSTM+임베딩 41.66**으로, 임베딩 결합 후 ARIMA를 앞서며 **12조건 중 7조건**에서 우세합니다. 특히 예측이 어려운 구간의 격차가 큽니다 — SBC C-4(Lumpy) ARIMA **247.05** vs LSTM+PCA 44.14, SBC C-3(Erratic) 84.37 vs 34.83. ARIMA는 안정 계열에서 강하지만 Erratic·Lumpy에서 MAPE가 폭발해 실무 적용이 어렵습니다. (ARIMA는 계열별 단변량이라 정적 임베딩이 절편에 흡수돼 결합 자체가 불가능하기도 합니다.)
- **임베딩별 median MAPE는 근소차:** PCA 41.91 · AE 42.10 · PatchTST 42.53 · TS2Vec 42.79 · FastDTW 42.88 · GAF-CNN 43.14. 조건별 Best는 FastDTW 4 · AE 2 · GAF-CNN 2 · PatchTST 2 · TS2Vec 1 · PCA 1.
- **11장 scheme (제품수 가중 WMAPE) — 양 type 모두 SBC 우세:** C(저변동) 38.18 vs 38.19(−0.01, 사실상 동률) · E(고변동) 46.72 vs 47.60(−0.88). 논문의 변동성↔scheme 대응(저변동→ML·고변동→SBC)이 **본 축소 실습에서는 재현되지 않았습니다.** ML 군집이 30:3·31:2로 심하게 불균형해 ML scheme이 제 역할을 못 한 것이 주된 원인입니다.
- **13장 변수 중요도:** **lag_1 지배**(C 68.9% · E 73.7%) → 논문 LAG1 top importance와 일치.
- **14장 Optuna (대표모델 LSTM+임베딩):** C 18.53→16.57(**10.6%↓**), E 31.38→31.26(0.4%↓) — 검증 13주 MAPE 개선. type별 best 하이퍼파라미터 상이(C: hidden 256·2층 / E: hidden 64·1층).

> **2026-07 재실행** — `embeddings.py`의 `except → PCA` 폴백에 가려 GAF-CNN·TS2Vec이 조용히 PCA 결과를 반환하던 버그를 수정(commit `b6574f0`)한 뒤 06~14장을 재실행한 결과입니다. 이 수정으로 06장 군집 선정이 `AE+KMeans`→`TS2Vec+KMeans`로, 11장 scheme 결론이 바뀌었습니다. 상세는 [`docs/실험_프레임워크.md` §4](docs/실험_프레임워크.md).

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
| `tuning.py` | **Optuna(TPE)** — Phase1 XGBoost·iTransformer(조건, val RMSE), 14장 **LSTM+임베딩**(type, val MAPE) |
| `phase_experiments.py` | **Phase1/2 엔진** (3-way, 튜닝, 재귀, 정규화) |
| `phase_analysis.py` | 11장 **제품수 가중 WMAPE**(`thesis_wmape_by_type`) |
| `metrics.py` | MAE/RMSE/MAPE/MASE/WMAPE |
| `feature_importance.py` | 13장 XGBoost 변수 중요도 (gain, 도구용) |

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

1. **논문 SOTA ≠ 축소 실습 Best** — 논문 iTransformer 강세는 12,661 SKU·3년의 대규모 맥락. 66 시계열에서는 iTransformer가 **하위권(51.6, 8위)** 이고 ARIMA(41.7)·LSTM(42.6)이 앞섭니다. 다만 **밀린 것은 딥러닝 전체가 아니라 트랜스포머 계열**입니다 — LSTM 2위·N-HiTS 4위로 XGBoost·RF보다 앞서고, 하위권은 파라미터가 많은 iTransformer·Autoformer 둘뿐입니다. 원인은 구현이 아니라 **학습 대상 규모**입니다: 딥러닝은 조건(type×cluster) 단위 global model이라 계열 수가 곧 표본 수인데 본 실습은 조건당 **1~19개**(중앙값 6)뿐입니다. 자세한 근거는 [`docs/실험_프레임워크.md` §3](docs/실험_프레임워크.md)을 참고하세요.
2. **SBC vs ML 우열은 데이터 규모에 의존** — 본 축소 실습(2-type·66 시계열)에서는 **양 type 모두 SBC 우세**로, 논문의 변동성↔scheme 대응(저변동→ML·고변동→SBC)이 재현되지 않았습니다. ML 군집이 30:3·31:2로 심하게 불균형해 ML scheme이 제 역할을 못 했습니다. 표본이 크고 변동 구조가 선명해야 방법론 우열이 드러납니다.
3. **MAPE는 간헐·near-zero 수요에서 불안정** — 롱테일 family의 MAPE는 크게 튈 수 있어 median·WMAPE를 함께 봄.
4. **단일 데이터셋** — Ecuador 2-type 결과가 모든 소매에 일반화되지는 않음.

---

## 라이선스

코드는 **연구·교육 목적**으로만 사용 가능합니다. 상업적 이용은 저자 사전 허가가 필요합니다.
