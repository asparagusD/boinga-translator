"""Tests for tools/build_notebook.py: tiny spec -> .ipynb layout, determinism and --check."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import build_notebook
import nbformat
import pytest

SPEC = '''"""Demo spec."""
from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C9", run=2, slug="demo", title="Demo run", branch="chunk/C9-demo",
    cells=[
        md("## Step one"),
        md("Explains the form."),
        form("obs", "Listening notes", {"VOICE": ["female", "male"], "OK": True}),
        md("Says hello."),
        code("hello", "Say hello", "print('PASS: hello')", expect=["^PASS: hello$"]),
    ],
    gpu={gpu},
)
'''


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake repo with the real bootstrap cells and one spec."""
    (tmp_path / "notebooks" / "specs").mkdir(parents=True)
    real = build_notebook.ROOT / "notebooks" / "colab_bootstrap.py"
    (tmp_path / "notebooks" / "colab_bootstrap.py").write_text(
        real.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "notebooks" / "specs" / "demo.py").write_text(
        SPEC.replace("{gpu}", "True"), encoding="utf-8")
    monkeypatch.setattr(build_notebook, "ROOT", tmp_path)
    return tmp_path


def _built(repo: Path) -> dict:
    assert build_notebook.main(["notebooks/specs/demo.py"]) == 0
    return json.loads((repo / "notebooks" / "runs" / "C9_demo_run2.ipynb").read_text("utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def test_layout_ids_and_markers(repo: Path) -> None:
    nb = _built(repo)
    ids = [c["metadata"].get("boinga", {}).get("id") for c in nb["cells"]]
    assert ids == [None, None, "BOOT.1", None, "BOOT.2", None, "BOOT.3", None, "BOOT.4",
                   None, "HDR", None, None, "hello", None, None, "obs"]
    for prev, cell in zip(nb["cells"], nb["cells"][1:], strict=False):
        if cell["cell_type"] == "code":
            assert prev["cell_type"] == "markdown"  # every cell has its explanation
    title = _src(nb["cells"][0])
    assert title.startswith("# C9 run 2 — Demo run") and "chunk/C9-demo" in title
    assert "colab_runs/C9_demo_run2.ipynb" in title
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cid = cell["metadata"]["boinga"]["id"]
            assert _src(cell).split("\n")[0].startswith(f"# [boinga {cid}] ")
            ast.parse(re.sub(r"^\s*[!%].*$", "pass", _src(cell), flags=re.MULTILINE))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert _src(nb["cells"][13]).endswith("cell_ok('hello')")
    assert "run_header('C9', 2)" in _src(nb["cells"][10])
    assert [_src(c) for c in nb["cells"][11:13]] == ["## Step one", "Says hello."]
    assert _src(nb["cells"][14]).startswith("## Your observations")
    assert _src(nb["cells"][15]) == "Explains the form."  # moved with its form
    nbformat.validate(nbformat.reads(json.dumps(nb), as_version=4))


def test_branch_baked_into_bootstrap(repo: Path) -> None:
    boot2 = _src(_built(repo)["cells"][4])
    assert 'BRANCH = "chunk/C9-demo"' in boot2
    assert 'BRANCH = "main"' not in boot2


def test_metadata_gpu_and_spec_hash(repo: Path) -> None:
    nb = _built(repo)
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["provenance"] == []
    meta = nb["metadata"]["boinga"]
    assert meta["chunk"] == "C9" and meta["run"] == 2 and meta["branch"] == "chunk/C9-demo"
    assert meta["spec"] == "notebooks/specs/demo.py" and re.fullmatch(r"[0-9a-f]{64}",
                                                                       meta["spec_sha256"])
    spec = repo / "notebooks" / "specs" / "demo.py"
    spec.write_text(SPEC.replace("{gpu}", "False"), encoding="utf-8")
    nb = _built(repo)
    assert "accelerator" not in nb["metadata"]
    assert all(not c["metadata"].get("boinga", {}).get("gpu") for c in nb["cells"])


def test_deterministic(repo: Path) -> None:
    out = repo / "notebooks" / "runs" / "C9_demo_run2.ipynb"
    _built(repo)
    first = out.read_bytes()
    _built(repo)
    assert out.read_bytes() == first
    assert b"\r\n" not in first


def test_check_detects_stale_and_orphans(repo: Path) -> None:
    assert build_notebook.main(["--all", "--check"]) == 1  # not built yet
    assert build_notebook.main(["--all"]) == 0
    assert build_notebook.main(["--all", "--check"]) == 0
    spec = repo / "notebooks" / "specs" / "demo.py"
    spec.write_text(spec.read_text("utf-8").replace("Say hello", "Say hi"), encoding="utf-8")
    assert build_notebook.main(["--all", "--check"]) == 1
    assert build_notebook.main(["notebooks/specs/demo.py", "--check"]) == 1
    assert build_notebook.main(["--all"]) == 0
    (repo / "notebooks" / "runs" / "C9_demo_run1.ipynb").write_text("{}", encoding="utf-8")
    assert build_notebook.main(["--all", "--check"]) == 1  # orphan: no spec builds it


def test_all_with_no_specs_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_notebook, "ROOT", tmp_path)
    assert build_notebook.main(["--all", "--check"]) == 0


def test_spec_without_notebook_rejected(repo: Path) -> None:
    (repo / "notebooks" / "specs" / "bad.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="NOTEBOOK"):
        build_notebook.main(["notebooks/specs/bad.py"])


def test_repo_notebooks_are_in_sync() -> None:
    assert build_notebook.main(["--all", "--check"]) == 0


def test_two_specs_building_the_same_notebook_rejected(repo: Path) -> None:
    spec = repo / "notebooks" / "specs" / "demo.py"
    (repo / "notebooks" / "specs" / "demo_copy.py").write_text(
        spec.read_text(encoding="utf-8"), encoding="utf-8")
    assert build_notebook.main(["--all"]) == 1


@pytest.mark.parametrize("cells", [
    'code("x", "X", "print(1)")',
    'md("About x."), code("x", "X", "print(1)"), code("y", "Y", "print(2)")',
    'md("About x."), code("x", "X", "print(1)"), form("f", "F", {"A": ""})',
])
def test_cell_without_explanation_rejected(repo: Path, cells: str) -> None:
    spec = repo / "notebooks" / "specs" / "demo.py"
    body = SPEC[SPEC.index("        md("):SPEC.index("    ],")]
    spec.write_text(SPEC.replace(body, f"        {cells},\n").replace("{gpu}", "True"),
                    encoding="utf-8")
    with pytest.raises(ValueError, match=r"needs an md\(\) explanation"):
        build_notebook.main(["notebooks/specs/demo.py"])


def test_output_override(repo: Path) -> None:
    spec = repo / "notebooks" / "specs" / "demo.py"
    spec.write_text(SPEC.replace("{gpu}", "True").replace(
        "gpu=True,", 'gpu=True, output="notebooks/demo_final.ipynb",'), encoding="utf-8")
    assert build_notebook.main(["notebooks/specs/demo.py"]) == 0
    out = repo / "notebooks" / "demo_final.ipynb"
    assert out.is_file() and not (repo / "notebooks" / "runs" / "C9_demo_run2.ipynb").exists()
    nb = json.loads(out.read_text("utf-8"))
    assert "`colab_runs/demo_final.ipynb`" in _src(nb["cells"][0])  # save name follows output
    assert build_notebook.main(["--all", "--check"]) == 0
    out.write_text("{}", encoding="utf-8")
    assert build_notebook.main(["--all", "--check"]) == 1  # stale override is caught too
