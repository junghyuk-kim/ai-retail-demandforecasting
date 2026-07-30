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

> 학위논문 실증은 국내 소매업체 물류센터 데이터로 **12,661 SKU·2센터**. 본 실습은 **66 시계열**의 교육용 축소판이므로, 표본이 작고 변동 스펙트럼이 좁아 방법론 간 성능 차이가 논문만큼 크지 않을 수 있습니다.

데이터는 라이선스상 저장소에 포함되지 않습니다. Kaggle에서 받아 `data/raw/`에 넣고 `02`장부터 실행하세요.

---

## 핵심 설계 원칙 (논문 방법론 반영)

소매 수요 예측에서 결과의 타당성을 좌우하는 아래 설계를 논문 방법론(§4.2·4.4·4.5)에 맞춰 반영했습니다.

| 원칙 | 내용 | 근거 |
|------|------|------|
| **DL 정규화** | iTransformer/Autoformer 입력을 열별 **Min-Max 정규화**(논문 §4.2) | 대규모 판매량(GROCERY I ~26만) 없이 트랜스포머 안정 학습(iTransformer median MAPE 33.8) |
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
→ 채택: **TS2Vec + KMeans, [61, 5]** (cluster2 = GROCERY I·BEVERAGES·CLEANING 등 **메가셀러**(주 평균 125,701), cluster1 = **롱테일**(4,888))

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

- **Phase1 10모델 median MAPE (31~39로 촘촘):** SBA 31.4 · N-HiTS 32.7 · ARIMA 32.9 · **iTransformer 33.8** · XGBoost 34.3 · LSTM 34.5 · TSB 34.6 · Prophet 36.5 · RF 37.2 · Autoformer 39.4. 축소 데이터라 단순 모델도 경쟁적이며 **iTransformer는 중위권**입니다. 조건별 승자는 **통계 7**(ARIMA 5·Prophet 2) · **딥러닝 3**(LSTM 2·N-HiTS 1) · **머신러닝 2**(XGBoost 1·RF 1).
- **median과 mean이 갈리는 지점 — 견고성:** mean MAPE로 보면 순위가 뒤집힙니다. **LSTM 45.0 · ARIMA 45.7** 이 압도적이고, median 1위였던 SBA는 **161.9**, TSB **174.0**, iTransformer **141.7** 로 폭발합니다. 즉 통계 모델은 **전형적인 계열에는 강하지만 어려운 계열에서 무너집니다**. LSTM만 median 34.5·mean 45.0으로 양쪽 모두 안정적입니다.
- **Phase2 base = LSTM + 임베딩:** ARIMA·SBA는 **계열별 단변량이라 임베딩 결합이 불가능**합니다(정적 임베딩이 절편에 흡수됨). 결합 가능한 모델 중 mean MAPE 최저인 **LSTM**을 base로 고정합니다. 임베딩 결합으로 median 34.5→**34.0**, mean 45.0→**44.3**(LSTM+GAF-CNN) 개선됩니다.
- **임베딩 6종은 근소차:** median 기준 GAF-CNN 34.0 · PCA 34.3 · AE 34.4 · FastDTW 34.5 · PatchTST 34.6 · TS2Vec 35.2. 조건별 Best는 FastDTW 4 · AE 2 · GAF-CNN 2 · PatchTST 2 · TS2Vec 1 · PCA 1.
- **11장 scheme (제품수 가중 WMAPE) — 양 type 모두 SBC 우세:** C(저변동) 38.18 vs 38.19(−0.01, 사실상 동률) · E(고변동) 46.72 vs 47.60(−0.88). 논문의 변동성↔scheme 대응(저변동→ML·고변동→SBC)이 **본 축소 실습에서는 재현되지 않았습니다.** ML 군집이 30:3·31:2로 심하게 불균형해 ML scheme이 제 역할을 못 한 것이 주된 원인입니다.
- **13장 변수 중요도:** **lag_1 지배**(C 68.9% · E 73.7%) → 논문 LAG1 top importance와 일치.
- **14장 Optuna (대표모델 LSTM+임베딩):** C 18.53→16.57(**10.6%↓**), E 31.38→31.26(0.4%↓) — 검증 13주 MAPE 개선. type별 best 하이퍼파라미터 상이(C: hidden 256·2층 / E: hidden 64·1층).

---

## 학위논문과 결론이 다른 이유 — 표본 규모가 알고리즘 선택을 바꾼다

학위논문은 국내 소매업체 물류센터의 **비공개 데이터**(연구 목적 한정)로 수행했습니다. 독자가 직접 재현할 수 있어야 하므로, 이 책은 의도적으로 **공개 데이터(Kaggle Favorita)** 로 같은 방법론을 다시 실행합니다. 두 데이터의 규모 차이는 다음과 같습니다.

| | 학위논문 (비공개) | 본 실습 (공개) | 배율 |
|---|---|---|---|
| 분석 단위 | SKU | 매장유형 × 상품군 | — |
| 계열 수 | **12,661 × 2센터** | **66** | 약 380배 |
| 기간 | 156주 (3년) | 242주 | 0.6배 |
| 총 레코드 | 약 395만 | 약 1.6만 | 약 247배 |
| 조건당 학습 계열 | 수백~1만 | **1~19** (중앙값 6) | — |

**결론이 갈리는 것은 오류가 아니라 이 표본 차이의 자연스러운 귀결입니다.** 그리고 그 자체가 실무에 유용한 발견입니다.

| 쟁점 | 학위논문 (대규모) | 본 실습 (소규모) |
|------|------------------|-----------------|
| **최적 알고리즘** | iTransformer 등 트랜스포머 SOTA | **10모델이 31~39로 촘촘**, 조건별 승자는 통계 7 / DL 3 / ML 2 |
| **iTransformer** | 계열 폭이 넓어 어텐션이 공통 구조를 학습 | 중위권(33.8) — 표본 부족으로 우위가 사라짐 |
| **scheme 대응** | 고변동→SBC · 저변동→ML | **양 type 모두 SBC 우세** (ML 군집이 30:3·31:2로 퇴화) |
| **임베딩 효과** | 클러스터링·예측 모두 기여 | 6종 median 34.0~35.2로 근소차 |

여기서 얻을 교훈은 세 가지입니다.

1. **"최신 SOTA"보다 보유 데이터의 계열 폭을 먼저 본다.** 딥러닝의 강점은 계열 길이가 아니라 다수 계열의 공통 패턴에서 나옵니다. 계열이 수십 개뿐이면 대형 트랜스포머의 우위가 사라집니다.
2. **median 하나로 판단하지 않는다.** SBA는 median 31.4로 1위지만 mean은 **161.9**로 최하위권입니다 — 전형적인 계열에는 강하고 어려운 계열에서 무너진다는 뜻입니다. 재고·발주에서는 최악 구간의 오차가 비용을 좌우하므로 **median과 mean을 함께** 봐야 합니다. 두 지표 모두 안정적인 모델은 LSTM(34.5 / 45.0)입니다.
3. **분류 체계(SBC vs ML)의 우열도 규모에 의존한다.** ML 클러스터링은 계열이 충분해야 의미 있는 군집을 만듭니다. 66계열에서는 30:3으로 퇴화해 규칙 기반 SBC가 안정적으로 우세했습니다.

> 같은 방법론·같은 공식 구현으로 두 데이터를 돌렸을 때 승자가 뒤바뀐다는 사실은, **알고리즘의 절대적 우열이라는 것은 없으며 데이터 규모가 선택을 좌우한다**는 점을 보여줍니다. 실무에서 모델을 고를 때 벤치마크 순위표보다 자기 데이터의 구조를 먼저 확인해야 하는 이유입니다.

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

python run_pipeline.py            # 02~14 전체 순차 실행
python run_pipeline.py 06 11      # 특정 구간만
python run_pipeline.py --list     # 실행 순서 확인
```

개별 장만 돌리려면 `python execute_notebook.py "06_클러스터링_방법.ipynb"` 를 사용합니다.
실행 로그는 `data/processed/log_<장>.txt` 에 남습니다.

> 07~10장은 결과를 `data/processed/*.parquet` 에 캐시합니다. 파라미터를 바꿔 다시 계산하려면 해당 캐시 파일을 지우고 실행하세요.

Python 3.10+, GPU 권장(DL·임베딩 가속). 자세한 방법론: [`docs/실험_프레임워크.md`](docs/실험_프레임워크.md)

---

## 해석 시 유의사항

1. **논문 SOTA ≠ 축소 실습 Best** — 논문 iTransformer 강세는 12,661 SKU·3년의 대규모 맥락입니다. 66 시계열에서는 10모델이 median MAPE **31~39로 촘촘**하고 iTransformer는 **중위권(33.8)** 에 머뭅니다. 딥러닝은 조건(type×cluster) 단위 global model이라 계열 수가 곧 표본 수인데, 본 실습은 조건당 **1~19개**(중앙값 6)뿐이라 계열 간 공유 표현을 학습할 여지가 적습니다. 자세한 근거는 [`docs/실험_프레임워크.md` §3](docs/실험_프레임워크.md)을 참고하세요.
2. **SBC vs ML 우열은 데이터 규모에 의존** — 본 축소 실습(2-type·66 시계열)에서는 **양 type 모두 SBC 우세**로, 논문의 변동성↔scheme 대응(저변동→ML·고변동→SBC)이 재현되지 않았습니다. ML 군집이 30:3·31:2로 심하게 불균형해 ML scheme이 제 역할을 못 했습니다. 표본이 크고 변동 구조가 선명해야 방법론 우열이 드러납니다.
3. **MAPE는 간헐·near-zero 수요에서 불안정** — 롱테일 family의 MAPE는 크게 튈 수 있어 median·WMAPE를 함께 봄.
4. **단일 데이터셋** — Ecuador 2-type 결과가 모든 소매에 일반화되지는 않음.

---

## 라이선스

코드는 **연구·교육 목적**으로만 사용 가능합니다. 상업적 이용은 저자 사전 허가가 필요합니다.
