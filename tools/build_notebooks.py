"""Build Colab .ipynb files from "# %%" cell files; the header comment becomes a markdown cell."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = {"notebooks/colab_bootstrap.py": "notebooks/c1_colab_bootstrap.ipynb"}  # source -> nb
CELL_RE = re.compile(r"^# %%.*$", re.MULTILINE)
METADATA = {
    "accelerator": "GPU",
    "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


def _source(text: str) -> list[str]:
    """Notebook-style source: list of lines, each keeping its newline except the last."""
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def build(py_text: str) -> str:
    """Return the .ipynb JSON for a "# %%" cell file (deterministic, LF, no outputs)."""
    head, *bodies = CELL_RE.split(py_text)
    titles = CELL_RE.findall(py_text)
    doc = ast.get_docstring(ast.parse(head)) or ""
    comments = [ln[2:] if ln.startswith("# ") else ln[1:] for ln in head.splitlines()
                if ln.startswith("#")]
    markdown = (f"# {doc}\n\n" + "\n".join(comments)).strip()
    cells = [{"cell_type": "markdown", "metadata": {}, "source": _source(markdown)}]
    for title, body in zip(titles, bodies, strict=True):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": _source(f"{title}\n{body.strip(chr(10))}")})
    nb = {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 0}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if any notebook is stale")
    args = ap.parse_args(argv)
    stale = []
    for src, dst in sorted(PAIRS.items()):
        expected = build((ROOT / src).read_text(encoding="utf-8"))
        target = ROOT / dst
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if args.check:
            if current != expected:
                stale.append(dst)
        else:
            target.write_text(expected, encoding="utf-8", newline="\n")
            print(f"wrote {dst}")
    if stale:
        print(f"stale notebooks: {', '.join(stale)}. Run `python tools/build_notebooks.py`.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
