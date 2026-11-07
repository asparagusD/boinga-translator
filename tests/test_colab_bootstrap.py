"""Tests for notebooks/colab_bootstrap.py: parses as Python, 4 cells, guards, no secrets in it."""

from __future__ import annotations

import ast
import re

import pytest

from boinga_translator import paths

BOOTSTRAP = paths.REPO / "notebooks" / "colab_bootstrap.py"
TOKEN_LIKE = re.compile(
    r"(ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]+"  # GitHub token prefixes
    r"|x-access-token:[A-Za-z0-9_]{8,}"  # a literal token in a credential URL
    r"|https://[A-Za-z0-9_]{20,}@github\.com"  # token embedded in a clone URL
)


@pytest.fixture(scope="module")
def text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _cells(text: str) -> list[str]:
    return re.split(r"^# %%.*$", text, flags=re.MULTILINE)[1:]


def test_parses_as_python(text: str) -> None:
    ast.parse(text)


def test_has_four_cells(text: str) -> None:
    assert len(re.findall(r"^# %%", text, flags=re.MULTILINE)) == 4
    for cell in _cells(text):
        ast.parse(cell)  # each cell must parse on its own (raises SyntaxError otherwise)


def test_no_hard_coded_token_like_strings(text: str) -> None:
    assert not TOKEN_LIKE.search(text)
    assert 'userdata.get("GH_TOKEN")' in text
    token_lines = [ln for ln in text.splitlines() if "{token}" in ln]
    assert len(token_lines) == 1 and "x-access-token:{token}@github.com" in token_lines[0]
    assert "print(token" not in text


def test_guards_and_branch(text: str) -> None:
    cells = _cells(text)
    assert "drive.mount" in cells[0]
    assert re.search(r'^BRANCH = "main"', cells[1], flags=re.MULTILINE)
    assert "/content/drive/MyDrive" in cells[1]  # cell 2 needs cell 1
    # Colab-made reports (tracked or untracked) must be dropped before switch/pull
    cleanup = cells[1].index('git("clean", "-fdq", "--", "reports/")')
    assert cells[1].index('git("checkout", "-q", "--", "reports/")') < cleanup
    assert cleanup < cells[1].index('git("switch", BRANCH)')
    for cell in cells[2:]:
        assert "FAIL: run the bootstrap cells first" in cell
    assert '".[train]"' in cells[2]
    assert "paths.ensure_dirs()" in cells[3]
    assert "env_report" not in cells[3]  # HDR (run_header) is the only env.json emitter
