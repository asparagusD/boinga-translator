"""Tests for the committed run notebooks: in sync, explained cells, C1 bootstrap shape, no secrets."""

from __future__ import annotations

import ast
import itertools
import json
import re
from pathlib import Path

import build_notebook
import pytest

from boinga_translator import paths

C1_NOTEBOOK = paths.REPO / "notebooks" / "runs" / "C1_bootstrap_run1.ipynb"
BUILT = sorted([*(paths.REPO / "notebooks" / "runs").glob("*.ipynb"),
                *(paths.REPO / "notebooks").glob("*.ipynb")])  # output= notebooks too


def test_committed_notebooks_are_in_sync() -> None:
    assert build_notebook.main(["--all", "--check"]) == 0, "run `tools/build_notebook.py --all`"


@pytest.mark.parametrize("nb_path", BUILT, ids=lambda p: p.name)
def test_run_notebooks_have_no_outputs_or_tokens(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert not re.search(r"ghp_|gho_|ghs_|github_pat_", json.dumps(nb))


@pytest.mark.parametrize("nb_path", BUILT, ids=lambda p: p.name)
def test_every_cell_has_an_explanation(nb_path: Path) -> None:
    """CLAUDE.md: every code/form cell comes right after a markdown cell explaining it."""
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
    for prev, cell in itertools.pairwise(cells):
        if cell["cell_type"] == "code":
            cid = cell["metadata"].get("boinga", {}).get("id")
            assert prev["cell_type"] == "markdown", f"{cid} has no explanation before it"
            assert "".join(prev["source"]).strip(), f"{cid}: empty explanation"


def test_c1_bootstrap_notebook_shape() -> None:
    nb = json.loads(C1_NOTEBOOK.read_text(encoding="utf-8"))
    ids = [c["metadata"].get("boinga", {}).get("id") for c in nb["cells"]]
    assert ids == [None, None, "BOOT.1", None, "BOOT.2", None, "BOOT.3", None, "BOOT.4",
                   None, "HDR"]
    assert nb["nbformat"] == 4 and nb["metadata"]["colab"]["gpuType"] == "T4"
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["boinga"]["branch"] == "main"
    for cell in nb["cells"][1:]:
        if cell["cell_type"] == "markdown":
            continue
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    assert 'BRANCH = "main"' in "".join(nb["cells"][4]["source"])
    assert not re.search(r"ghp_|github_pat_", json.dumps(nb))
