# AI와 소매 수요 예측 — 실습 저장소

Ecuador Favorita Store Sales로 소매 수요 예측 전 과정(패턴 분석 → 분류·클러스터링 → 알고리즘 벤치마크 → 임베딩 하이브리드 → scheme 비교)을 재현하는 Jupyter 실습입니다.

관련 도서: **AI와 소매 수요 예측** (김정혁, 커뮤니케이션북스)

---

## 데이터·실험 단위

| 항목 | 내용 |
|------|------|
| 출처 | [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
| 원 단위 | store type 5 × family 33 = **165** 주간 시계열 |
| 실험 대상 | System-Level CV로 선정한 **2-type** — 고변동 **E**(0.429) · 저변동 **C**(0.259) → **66** 시계열 |
| 분할 | train ≤201707 (216주) / val 201708–201720 (13주) / test 201721–201733 (13주) |
| 예측 지평 | 13주 (lookback 26) |

학위논문 실증은 국내 소매 **12,661 SKU·2센터**. 본 실습은 공개 데이터 **66 시계열** 축소판이라, type 간 변동 차이와 방법론 간 성능 차이가 논문만큼 크지 않을 수 있습니다.

데이터는 `data/raw/`에 넣은 뒤 `02`장부터 실행하세요.

---

## 알고리즘·클러스터

| 구분 | 내용 |
|------|------|
| 통계 (07) | ARIMA, Prophet, SBA, TSB |
| ML (08) | Random Forest, XGBoost |
| DL (09) | LSTM, Autoformer, N-HiTS, iTransformer (공식 구현) |
| 임베딩 (06·10) | PCA, FastDTW, AE, GAF-CNN, TS2Vec, PatchTST |
| SBC (05) | ADI 1.32 · CV² 0.49 → Smooth / Intermittent / Erratic / Lumpy |
| ML 군집 (06) | **TS2Vec + KMeans, K=2** → [61, 5] (롱테일 / 메가셀러) |

DL·임베딩 공식 코드는 `vendor/` submodule입니다.

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

평가: 07–10장은 **MAPE**(조건별 Best), 11장은 제품수 가중 **WMAPE**, 06장은 Silhouette·DB.

---

## 노트북 흐름

```
02 전처리·2-type → 03 피처 → 05 SBC → 06 ML클러스터
→ 07 통계 → 08 ML → 09 DL → 10 임베딩 → 11 scheme 비교 → 12 RIDR
→ 13 변수중요도 → 14 Optuna
```

| 장 | 내용 |
|----|------|
| 02–04 | 주간 집계, 2-type 선정, 피처, EDA |
| 05–06 | SBC 4분류 / ML 군집 (TS2Vec+KMeans) |
| 07–09 | **12조건**(SBC 8 + ML 4) × 10모델 |
| 10 | **LSTM × 6임베딩** (72조합) |
| 11 | type별 SBC vs ML 가중 WMAPE |
| 12–14 | RIDR·변수 중요도·Optuna |

---

## 주요 결과

- **Phase1 (median MAPE):** SBA 31.4 · N-HiTS 32.7 · ARIMA 32.9 · iTransformer 33.8 · XGBoost 34.3 · LSTM 34.5 … (31~39로 좁음). 조건별 승자: 통계 7 · DL 3 · ML 2.
- **median vs mean:** SBA는 median 1위지만 mean **161.9**. LSTM은 median 34.5 · mean **45.0**으로 둘 다 안정적.
- **Phase2 base = LSTM:** ARIMA 등은 단변량이라 임베딩 결합이 불가. 결합 가능한 모델 중 mean MAPE 최저인 LSTM을 고정. 임베딩 후 median 34.5→**34.0**, mean 45.0→**44.3**(LSTM+GAF-CNN). 6종 간 차이는 작음.
- **11장 scheme:** C 38.18 vs 38.19 · E 46.72 vs 47.60 → **둘 다 SBC**. 논문의 저변동→ML·고변동→SBC 대응은 이 축소 표본에서 재현되지 않음. ML 군집이 C 30:3 · E 31:2로 치우친 점이 큼.
- **13–14장:** lag_1 지배(C 68.9% · E 73.7%). Optuna로 C 18.53→16.57, E 31.38→31.26 (val MAPE).

방법론 상세: [`docs/실험_프레임워크.md`](docs/실험_프레임워크.md)

---

## 실행

```bash
pip install -r requirements.txt
git submodule update --init --recursive
cd code
python run_pipeline.py            # 02~14
python run_pipeline.py 06 11      # 구간만
```

개별 장: `python execute_notebook.py "06_클러스터링_방법.ipynb"`  
캐시: `data/processed/*.parquet` — 재계산 시 해당 파일 삭제 후 실행.

Python 3.10+, GPU 권장.

---

## 라이선스

연구·교육 목적. 상업적 이용은 저자 사전 허가 필요.
