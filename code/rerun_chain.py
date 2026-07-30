"""임베딩 버그 수정 후 영향 체인 재실행.

GAF-CNN·TS2Vec 이 조용히 PCA 로 대체되던 버그(commit b6574f0)를 고친 뒤,
그 결과가 반영되어야 하는 노트북을 순서대로 실행한다.

영향 경로
--------
06 클러스터링   임베딩 6종 × 클러스터링 4종 그리드로 ML scheme 결정
                → 수정 후 선정이 AE+KMeans 에서 TS2Vec+KMeans 로 바뀜
07·08·09        Phase1 (ML 클러스터 배정이 바뀌었으므로 재계산)
10              임베딩 기반 시계열 분석
11              하이브리드 (Phase2 = Phase1 best + 임베딩)
12·13·14        군집·최종모델에 의존하는 후속 분석

사용법:
    python rerun_chain.py            # 07부터 14까지 (06은 이미 완료)
    python rerun_chain.py 10 11      # 일부만
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
NB = {
    "06": "06_클러스터링_방법.ipynb",
    "07": "07_통계_예측모델.ipynb",
    "08": "08_머신러닝_예측모델.ipynb",
    "09": "09_딥러닝_예측모델.ipynb",
    "10": "10_임베딩_기반_시계열분석.ipynb",
    "11": "11_하이브리드_수요예측.ipynb",
    "12": "12_제품_수요패턴_RIDR_분석.ipynb",
    "13": "13_변수_중요도_분석.ipynb",
    "14": "14_Optuna_튜닝.ipynb",
}


def run(key: str) -> bool:
    nb = NB[key]
    log = CODE.parent / "data" / "processed" / f"log_rerun_{key}.txt"
    print(f"\n{'=' * 60}\n[{key}] {nb}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "execute_notebook.py", nb],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=str(CODE))
    el = (time.time() - t0) / 60
    ok = r.returncode == 0
    print(f"[{key}] {'완료' if ok else '실패(exit %d)' % r.returncode} — {el:.1f}분", flush=True)
    if not ok:
        print(f"  로그: {log}", flush=True)
    return ok


if __name__ == "__main__":
    keys = sys.argv[1:] or ["07", "08", "09", "10", "11", "12", "13", "14"]
    t0 = time.time()
    for k in keys:
        if not run(k):
            print(f"\n[{k}] 에서 중단.")
            sys.exit(1)
    print(f"\n{'=' * 60}\n전체 완료 — {(time.time() - t0) / 60:.1f}분")
