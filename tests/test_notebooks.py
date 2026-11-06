"""Tests for tools/build_notebooks.py: committed .ipynb files match their "# %%" sources."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import build_notebooks
import pytest

from boinga_translator import paths


@pytest.mark.parametrize(("src", "dst"), sorted(build_notebooks.PAIRS.items()))
def test_committed_notebook_is_in_sync(src: str, dst: str) -> None:
    expected = build_notebooks.build((paths.REPO / src).read_text(encoding="utf-8"))
    committed = (paths.REPO / dst).read_text(encoding="utf-8")
    assert committed == expected, f"{dst} is stale: run `python tools/build_notebooks.py`"


def test_c1_notebook_shape() -> None:
    nb = json.loads(build_notebooks.build(
        (paths.REPO / "notebooks" / "colab_bootstrap.py").read_text(encoding="utf-8")))
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds == ["markdown", "code", "code", "code", "code"]
    assert nb["nbformat"] == 4 and nb["metadata"]["colab"]["gpuType"] == "T4"
    assert nb["metadata"]["accelerator"] == "GPU"
    for cell in nb["cells"][1:]:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    assert "".join(nb["cells"][1]["source"]).startswith("# %% Cell 1/4")
    assert not re.search(r"ghp_|github_pat_", json.dumps(nb))


def test_build_splits_header_and_cells() -> None:
    text = '"""Demo."""\n\n# Intro line\n\n# %% A\nx = 1\n\n# %% B\ny = 2\n'
    nb = json.loads(build_notebooks.build(text))
    assert "".join(nb["cells"][0]["source"]) == "# Demo.\n\nIntro line"
    assert ["".join(c["source"]) for c in nb["cells"][1:]] == ["# %% A\nx = 1", "# %% B\ny = 2"]


def test_check_mode_detects_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.py").write_text('"""A."""\n# %% one\nx = 1\n', encoding="utf-8")
    monkeypatch.setattr(build_notebooks, "ROOT", tmp_path)
    monkeypatch.setattr(build_notebooks, "PAIRS", {"a.py": "a.ipynb"})
    assert build_notebooks.main(["--check"]) == 1
    assert build_notebooks.main([]) == 0
    assert build_notebooks.main(["--check"]) == 0
