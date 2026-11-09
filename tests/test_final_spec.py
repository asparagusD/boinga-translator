"""Tests for notebooks/specs/final.py: 17 numbered sections, real `boinga` commands, flag cells."""

from __future__ import annotations

import re

import build_notebook
import nbkit
import pytest

from boinga_translator import cli, paths

SPEC = paths.REPO / "notebooks" / "specs" / "final.py"
HEADING = re.compile(r"^## (\d+)\. \S")
TRAINING = {"S10", "S10.train", "S11", "S12", "S12.play", "S13"}


@pytest.fixture(scope="module")
def notebook() -> nbkit.Notebook:
    return build_notebook.load_spec(SPEC)


def test_identity(notebook: nbkit.Notebook) -> None:
    assert (notebook.chunk, notebook.run, notebook.slug) == ("C12", 1, "final")
    assert notebook.branch == "main" and notebook.gpu  # after the C12 merge
    assert build_notebook.output_path(notebook) == "notebooks/boinga_translator.ipynb"


def test_seventeen_numbered_sections(notebook: nbkit.Notebook) -> None:
    numbers = [int(m.group(1)) for c in notebook.cells if c.kind == "markdown"
               for m in [HEADING.match(c.source)] if m]
    assert numbers == list(range(1, 18))


def test_cells_are_in_their_sections(notebook: nbkit.Notebook) -> None:
    section = 0
    for cell in notebook.cells:
        m = HEADING.match(cell.source) if cell.kind == "markdown" else None
        if m:
            section = int(m.group(1))
        elif cell.kind != "markdown":
            assert re.match(rf"^S{section}(\.|$)", str(cell.id)), (cell.id, section)
            assert cell.id == "S0" or cell.expect or "play" in str(cell.id) \
                or cell.allow_interrupt, f"{cell.id} has no expect"


def test_every_boinga_command_exists(notebook: nbkit.Notebook) -> None:
    sub = next(a for a in cli.build_parser()._actions if a.dest == "command")
    commands = set(sub.choices)
    used = [m.group(1) for c in notebook.checked_cells
            for m in re.finditer(r"^\s*!boinga ([a-z][a-z-]*)", c.source, re.MULTILINE)]
    assert used and not set(used) - commands, set(used) - commands
    tools = [m.group(1) for c in notebook.checked_cells
             for m in re.finditer(r"^\s*!python (tools/\S+\.py)", c.source, re.MULTILINE)]
    assert tools and all((paths.REPO / t).is_file() for t in tools), tools


def test_boinga_arguments_parse(notebook: nbkit.Notebook) -> None:
    parser = cli.build_parser()
    for cell in notebook.checked_cells:
        for m in re.finditer(r"^\s*!boinga (.+)$", cell.source, re.MULTILINE):
            parser.parse_args(m.group(1).split())  # SystemExit on a bad argument


def test_training_cells_are_behind_the_flag(notebook: nbkit.Notebook) -> None:
    flagged = {str(c.id) for c in notebook.checked_cells if c.skip_unless}
    assert flagged == TRAINING
    assert all(c.skip_unless == "RUN_TRAINING" for c in notebook.checked_cells if c.skip_unless)
    flag = notebook.cell("S0")
    assert 'RUN_TRAINING = False  #@param {type:"boolean"}' in flag.source
    first_flagged = min(i for i, c in enumerate(notebook.checked_cells) if c.skip_unless)
    assert notebook.checked_cells.index(flag) < first_flagged
    assert not any("promote" in c.source for c in notebook.checked_cells)  # stays manual


def test_live_demo_is_the_last_but_one_cell(notebook: nbkit.Notebook) -> None:
    ids = [c.id for c in notebook.checked_cells]
    assert ids[-2:] == ["S16.live", "S17"] and notebook.cell("S16.live").allow_interrupt
