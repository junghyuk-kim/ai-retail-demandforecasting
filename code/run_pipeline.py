"""노트북 파이프라인 순차 실행.

02~14장을 의존 순서대로 실행한다. 특정 구간만 다시 돌릴 수도 있다.

사용법
------
    python run_pipeline.py              # 02~14 전체
    python run_pipeline.py 06 11        # 06~11 구간만
    python run_pipeline.py --list       # 실행 순서 확인

각 장의 로그는 `data/processed/log_<장>.txt` 에 저장된다.

주의: 07~10장은 결과를 `data/processed/*.parquet` 에 캐시한다. 파라미터를
바꿔 다시 계산하려면 해당 캐시 파일을 지우고 실행한다.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent

ORDER = [
    ("02", "02_데이터셋_로드_전처리.ipynb", "일→주 집계, 고/저변동 2-type 선정"),
    ("03", "03_특징_엔지니어링.ipynb", "lag·rolling 등 패널 피처"),
    ("04", "04_수요패턴_분석.ipynb", "변동성·SBC 미리보기 (선택)"),
    ("05", "05_rule_base_분류.ipynb", "SBC 4분류 라벨"),
    ("06", "06_클러스터링_방법.ipynb", "임베딩×클러스터링 그리드 → ML 클러스터"),
    ("07", "07_통계_예측모델.ipynb", "Phase1 통계 4종"),
    ("08", "08_머신러닝_예측모델.ipynb", "Phase1 머신러닝 2종"),
    ("09", "09_딥러닝_예측모델.ipynb", "Phase1 딥러닝 4종"),
    ("10", "10_임베딩_기반_시계열분석.ipynb", "Phase2 LSTM×임베딩 6종"),
    ("11", "11_하이브리드_수요예측.ipynb", "SBC vs ML scheme 비교"),
    ("12", "12_제품_수요패턴_RIDR_분석.ipynb", "CV·RIDR 수요 밴드"),
    ("13", "13_변수_중요도_분석.ipynb", "XGBoost 변수 중요도"),
    ("14", "14_Optuna_튜닝.ipynb", "대표모델 하이퍼파라미터 튜닝"),
]


def run(key: str, nb: str, desc: str) -> bool:
    log = CODE.parent / "data" / "processed" / f"log_{key}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\n[{key}] {desc}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "execute_notebook.py", nb],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=str(CODE))
    el = (time.time() - t0) / 60
    ok = r.returncode == 0
    print(f"[{key}] {'완료' if ok else '실패 (exit %d)' % r.returncode} — {el:.1f}분", flush=True)
    if not ok:
        print(f"      로그: {log}", flush=True)
    return ok


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for k, nb, d in ORDER:
            print(f"  {k}  {d:38} {nb}")
        return 0

    lo = args[0] if args else ORDER[0][0]
    hi = args[1] if len(args) > 1 else ORDER[-1][0]
    todo = [x for x in ORDER if lo <= x[0] <= hi]
    if not todo:
        print(f"실행 대상이 없습니다: {lo}~{hi}")
        return 1

    t0 = time.time()
    for key, nb, desc in todo:
        if not run(key, nb, desc):
            print(f"\n[{key}] 에서 중단했습니다.")
            return 1
    print(f"\n{'=' * 60}\n{todo[0][0]}~{todo[-1][0]} 완료 — {(time.time() - t0) / 60:.1f}분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
