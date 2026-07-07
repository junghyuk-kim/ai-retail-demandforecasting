"""Execute a notebook and fix nbformat v5 output schema (stream name, metadata)."""
from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

import nbformat
from nbformat.validator import normalize


def fix_outputs(nb) -> None:
    for cell in nb.cells:
        if not getattr(cell, "id", None):
            cell.id = str(uuid.uuid4())[:8]
        ec = cell.get("execution_count")
        for out in cell.get("outputs", []):
            ot = out.get("output_type")
            if ot == "stream" and "name" not in out:
                text = "".join(out.get("text", []))
                out["name"] = (
                    "stderr"
                    if any(k in text.lower() for k in ("warning", "error", "traceback", "futurewarning"))
                    else "stdout"
                )
            if ot in ("execute_result", "display_data"):
                if "metadata" not in out:
                    out["metadata"] = {}
            if ot == "execute_result":
                if "execution_count" not in out and ec is not None:
                    out["execution_count"] = ec


def fix_notebook(path: Path) -> None:
    """Repair nbformat v5 output schema without re-executing."""
    path = path.resolve()
    _, nb = normalize(nbformat.read(path, as_version=4))
    fix_outputs(nb)
    _, nb = normalize(nb)
    nb.metadata.setdefault("language_info", {}).setdefault("name", "python")
    nb.metadata.setdefault("kernelspec", {}).setdefault("name", "python3")
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print("Fixed:", path)


def execute_notebook(path: Path, timeout: int = 7200) -> None:
    path = path.resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(path),
            f"--ExecutePreprocessor.timeout={timeout}",
        ],
        check=True,
        cwd=path.parent,
    )
    _, nb = normalize(nbformat.read(path, as_version=4))
    fix_outputs(nb)
    _, nb = normalize(nb)
    nb.metadata.setdefault("language_info", {}).setdefault("name", "python")
    nb.metadata.setdefault("kernelspec", {}).setdefault("name", "python3")
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print("Executed and fixed:", path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python execute_notebook.py <notebook.ipynb> [timeout_sec|--fix-only]")
    nb_path = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--fix-only":
        fix_notebook(nb_path)
    else:
        t = int(sys.argv[2]) if len(sys.argv) > 2 else 7200
        execute_notebook(nb_path, t)
