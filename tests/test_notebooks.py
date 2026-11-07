"""Tests for the committed run notebooks: in sync with their specs; C1 bootstrap shape, no secrets."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import build_notebook
import pytest

from boinga_translator import paths

C1_NOTEBOOK = paths.REPO / "notebooks" / "runs" / "C1_bootstrap_run1.ipynb"


def test_committed_notebooks_are_in_sync() -> None:
    assert build_notebook.main(["--all", "--check"]) == 0, "run `tools/build_notebook.py --all`"


@pytest.mark.parametrize("nb_path", sorted((paths.REPO / "notebooks" / "runs").glob("*.ipynb")),
                         ids=lambda p: p.name)
def test_run_notebooks_have_no_outputs_or_tokens(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert not re.search(r"ghp_|gho_|ghs_|github_pat_", json.dumps(nb))


def test_c1_bootstrap_notebook_shape() -> None:
    nb = json.loads(C1_NOTEBOOK.read_text(encoding="utf-8"))
    ids = [c["metadata"].get("boinga", {}).get("id") for c in nb["cells"]]
    assert ids == [None, "BOOT.1", "BOOT.2", "BOOT.3", "BOOT.4", "HDR"]
    assert nb["nbformat"] == 4 and nb["metadata"]["colab"]["gpuType"] == "T4"
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["boinga"]["branch"] == "main"
    for cell in nb["cells"][1:]:
        ast.parse("".join(cell["source"]))
        assert cell["outputs"] == [] and cell["execution_count"] is None
    assert 'BRANCH = "main"' in "".join(nb["cells"][2]["source"])
    assert not re.search(r"ghp_|github_pat_", json.dumps(nb))
