"""Helpers for importing vendor repos without shadowing code/utils."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]


def purge_vendor_modules(extra: tuple[str, ...] = ()) -> None:
    prefixes = ("models", "utils", "data", "src", *extra)
    for key in list(sys.modules):
        if key in prefixes or any(key.startswith(f"{p}.") for p in prefixes):
            mod = sys.modules.get(key)
            fn = getattr(mod, "__file__", "") or ""
            norm = fn.replace("\\", "/")
            if "/vendor/" in norm:
                del sys.modules[key]


def restore_project_utils() -> None:
    """Re-bind `utils` to this repo's package after vendor imports."""
    purge_vendor_modules()
    code_root = str(_CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    utils_pkg = sys.modules.get("utils")
    pkg_path = str(_CODE_ROOT / "utils")
    if utils_pkg is None or not getattr(utils_pkg, "__path__", None):
        if "utils" in sys.modules:
            del sys.modules["utils"]
        importlib.import_module("utils")
