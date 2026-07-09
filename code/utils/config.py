"""실험 대상 type(=논문의 센터) 선정 결과를 저장·로드하는 설정 모듈.

02장에서 System-Level CV 기준으로 고변동/저변동 type을 자동 선정해 JSON으로 저장하고,
이후 03~14장은 `selected_type_list()`로 해당 2개 type만 필터링해 실험 규모를 축소한다.
(논문의 2센터 A/B 비교 구조에 대응)
"""
from __future__ import annotations

import json

from .paths import DATA_PROCESSED

SELECTED_TYPES_JSON = DATA_PROCESSED / "selected_types.json"

# JSON 부재 시 폴백 — 02장에서 데이터 기반으로 선정된 값과 동일 (고변동 E / 저변동 C)
DEFAULT_SELECTED = {"high": "E", "low": "C"}


def save_selected_types(high: str, low: str, extra: dict | None = None) -> dict:
    """선정 결과를 JSON으로 저장. high=고변동, low=저변동."""
    payload = {"high": str(high), "low": str(low)}
    if extra:
        payload.update(extra)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    SELECTED_TYPES_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_selected_types() -> dict:
    """저장된 선정 결과 로드. 없으면 DEFAULT_SELECTED 반환."""
    if SELECTED_TYPES_JSON.exists():
        return json.loads(SELECTED_TYPES_JSON.read_text(encoding="utf-8"))
    return dict(DEFAULT_SELECTED)


def selected_type_list() -> list[str]:
    """실험 대상 type 리스트 [고변동, 저변동] — 다운스트림 필터용."""
    sel = load_selected_types()
    return [sel["high"], sel["low"]]


def filter_selected_types(df, type_col: str = "type"):
    """DataFrame을 선정된 2개 type으로 필터링."""
    return df[df[type_col].isin(selected_type_list())].copy()
