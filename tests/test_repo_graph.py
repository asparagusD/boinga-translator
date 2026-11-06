"""Tests for tools/repo_graph.py: node kinds, edge types, stability and --check on a fake repo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import repo_graph

NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Train\n"]},
        {"cell_type": "code", "source": ["%cd /content\n", "from boinga_translator import loader\n"]},
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}

FILES = {
    "src/boinga_translator/__init__.py": '"""Package."""\n',
    "src/boinga_translator/paths.py": '"""Paths."""\nCHAR_MAP = "configs/char_map.json"\n',
    "src/boinga_translator/loader.py": (
        '"""Load data."""\nfrom boinga_translator.paths import CHAR_MAP\n'
    ),
    "src/boinga_translator/nodoc.py": "X = 1\n",
    "tests/test_loader.py": '"""Loader test."""\nfrom boinga_translator import loader\n',
    "configs/char_map.json": "{}\n",
    "notebooks/train.ipynb": json.dumps(NOTEBOOK),
}


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def _nodes(root: Path) -> dict[str, dict]:
    return {n["path"]: n for n in repo_graph.build_graph(root)["nodes"]}


def test_nodes_and_kinds(fake_repo: Path) -> None:
    kinds = {path: n["kind"] for path, n in _nodes(fake_repo).items()}
    assert kinds == {
        "src/boinga_translator/__init__.py": "module",
        "src/boinga_translator/paths.py": "module",
        "src/boinga_translator/loader.py": "module",
        "src/boinga_translator/nodoc.py": "module",
        "tests/test_loader.py": "test",
        "configs/char_map.json": "config",
        "notebooks/train.ipynb": "notebook",
    }


def test_purposes_and_no_docstring(fake_repo: Path) -> None:
    nodes = _nodes(fake_repo)
    assert nodes["src/boinga_translator/nodoc.py"]["purpose"] == "(no docstring)"
    assert nodes["src/boinga_translator/loader.py"]["purpose"] == "Load data."
    assert nodes["notebooks/train.ipynb"]["purpose"] == "Train"


def test_four_edge_types(fake_repo: Path) -> None:
    edges = {(e["source"], e["target"], e["kind"]) for e in repo_graph.build_graph(fake_repo)["edges"]}
    assert edges == {
        ("src/boinga_translator/loader.py", "src/boinga_translator/paths.py", "import"),
        ("tests/test_loader.py", "src/boinga_translator/loader.py", "test"),
        ("notebooks/train.ipynb", "src/boinga_translator/loader.py", "notebook"),
        ("src/boinga_translator/paths.py", "configs/char_map.json", "config"),
    }


def test_output_is_stable(fake_repo: Path) -> None:
    json_path = fake_repo / repo_graph.JSON_OUT
    assert repo_graph.main(["--root", str(fake_repo)]) == 0
    first = json_path.read_bytes()
    assert repo_graph.main(["--root", str(fake_repo)]) == 0
    assert json_path.read_bytes() == first
    assert repo_graph.JSON_OUT not in _nodes(fake_repo)  # outputs never map themselves


def test_check_fails_after_file_added(
    fake_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert repo_graph.main(["--root", str(fake_repo)]) == 0
    assert repo_graph.main(["--root", str(fake_repo), "--check"]) == 0
    (fake_repo / "src/boinga_translator/new.py").write_text('"""New."""\n', encoding="utf-8")
    assert repo_graph.main(["--root", str(fake_repo), "--check"]) == 1
    assert "stale" in capsys.readouterr().err
