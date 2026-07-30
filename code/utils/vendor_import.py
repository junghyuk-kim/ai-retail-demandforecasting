"""Helpers for importing vendor repos without shadowing code/utils."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]

# 벤더 import 동안 잠시 치워둔 프로젝트 자체 모듈 (restore_project_utils에서 복구)
_STASHED: dict = {}


def purge_vendor_modules(extra: tuple[str, ...] = ()) -> None:
    """Drop cached vendor modules so a different vendor repo can be imported.

    Also stashes THIS repo's own top-level packages that share a name with a
    vendor module (notably `utils`). Vendored code such as ts2vec.py does
    `from utils import take_per_row`; if `sys.modules["utils"]` still points at
    code/utils, that import resolves to the wrong package and raises ImportError
    — which the callers used to swallow, silently degrading to PCA.
    """
    prefixes = ("models", "utils", "data", "src", *extra)
    for key in list(sys.modules):
        if key in prefixes or any(key.startswith(f"{p}.") for p in prefixes):
            mod = sys.modules.get(key)
            fn = getattr(mod, "__file__", "") or ""
            norm = fn.replace("\\", "/")
            if "/vendor/" in norm:
                del sys.modules[key]
            elif "/code/utils" in norm or norm.endswith("/code/utils/__init__.py"):
                # project package shadowing a vendor module name — stash it
                _STASHED[key] = sys.modules.pop(key)
            elif not fn:
                # namespace package (no __file__), e.g. a bare C:/Users/kjh/models
                # directory picked up from cwd. It has no submodules of its own but
                # blocks the vendor's regular package from being imported.
                _STASHED[key] = sys.modules.pop(key)


def restore_project_utils() -> None:
    """Re-bind `utils` to this repo's package after vendor imports."""
    # 벤더가 남긴 동명 모듈 제거 후, 치워뒀던 프로젝트 모듈을 되돌린다
    for key in list(sys.modules):
        mod = sys.modules.get(key)
        fn = (getattr(mod, "__file__", "") or "").replace("\\", "/")
        if "/vendor/" in fn and (key in ("models", "utils", "data", "src")
                                 or any(key.startswith(f"{p}.")
                                        for p in ("models", "utils", "data", "src"))):
            del sys.modules[key]
    while _STASHED:
        key, mod = _STASHED.popitem()
        sys.modules[key] = mod

    code_root = str(_CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    utils_pkg = sys.modules.get("utils")
    if utils_pkg is None or not getattr(utils_pkg, "__path__", None):
        if "utils" in sys.modules:
            del sys.modules["utils"]
        importlib.import_module("utils")
