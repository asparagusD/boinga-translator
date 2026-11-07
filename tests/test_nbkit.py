"""Tests for tools/nbkit.py: md/code/form cell sources, metadata, validation and Notebook."""

from __future__ import annotations

import ast

import nbkit
import pytest

CELL_OK = "from boinga_translator.colab_runtime import cell_ok; cell_ok('{}')"


def _lines(cell: nbkit.Cell) -> list[str]:
    return cell.source.split("\n")


def test_md_cell() -> None:
    cell = nbkit.md("  # Title\n  some text\n")
    assert cell.kind == "markdown" and cell.id is None
    assert cell.source == "# Title\nsome text"
    assert cell.metadata == {}


def test_code_marker_dedented_src_and_cell_ok() -> None:
    cell = nbkit.code("env", "Record the env", """
        import os
        if os.sep:
            print("PASS: ok")
    """, expect=[r"^PASS: ok$"])
    lines = _lines(cell)
    assert lines[0] == "# [boinga env] Record the env"
    assert lines[1:4] == ["import os", "if os.sep:", '    print("PASS: ok")']
    assert lines[-1] == CELL_OK.format("env")
    ast.parse(cell.source)


def test_code_forbid_always_has_defaults_and_metadata_mirrors_spec() -> None:
    cell = nbkit.code("x1", "T", "pass", forbid=["oops", "Traceback"], gpu=True, timeout_min=5)
    assert cell.forbid == (r"^FAIL:", "Traceback", "oops")
    assert cell.metadata == {"boinga": {
        "id": "x1", "kind": "code", "expect": [], "forbid": [r"^FAIL:", "Traceback", "oops"],
        "gpu": True, "timeout_min": 5}}


def test_shell_line_requires_expect() -> None:
    with pytest.raises(ValueError, match="shell"):
        nbkit.code("sh", "Shell", "!pip install -q foo")
    with pytest.raises(ValueError, match="shell"):
        nbkit.code("sh", "Shell", "if True:\n    !ls")
    cell = nbkit.code("sh", "Shell", "!ls && echo PASS: listed", expect=["^PASS: listed"])
    assert cell.expect == ("^PASS: listed",)


@pytest.mark.parametrize("bad_id", ["", "has space", "a'b", "BOOT.1", "HDR", "-x"])
def test_bad_or_reserved_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValueError):
        nbkit.code(bad_id, "T", "pass")


def test_bad_regex_and_multiline_title_rejected() -> None:
    with pytest.raises(ValueError, match="regex"):
        nbkit.code("a", "T", "pass", expect=["("])
    with pytest.raises(ValueError, match="title"):
        nbkit.code("a", "two\nlines", "pass")


def test_form_fields_parse_and_record_notes() -> None:
    cell = nbkit.form("obs", "Your observations", {
        "VOICE": ["female", "male"],
        "NOTES": "none",
        "SCORE": 3,
        "CLEAR": False,
    })
    lines = _lines(cell)
    assert lines[0] == "# [boinga obs] Your observations"
    assert lines[1] == "#@title Your observations"
    assert lines[2:6] == [
        'VOICE = "female" #@param ["female", "male"]',
        'NOTES = "none" #@param {type:"string"}',
        'SCORE = 3 #@param {type:"integer"}',
        'CLEAR = False #@param {type:"boolean"}',
    ]
    assert lines[6] == ("from boinga_translator.colab_runtime import record_notes; record_notes("
                        "'obs', {\"VOICE\": VOICE, \"NOTES\": NOTES, \"SCORE\": SCORE, "
                        "\"CLEAR\": CLEAR})")
    assert lines[-1] == CELL_OK.format("obs")
    assert cell.kind == "form" and cell.forbid == nbkit.ALWAYS_FORBID
    tree = ast.parse(cell.source)
    assigned = [n.targets[0].id for n in tree.body if isinstance(n, ast.Assign)]
    assert assigned == ["VOICE", "NOTES", "SCORE", "CLEAR"]


def test_form_after_src_and_quoting() -> None:
    cell = nbkit.form("f", "T", {"TEXT": 'say "hi"\\now'}, after_src="print(TEXT)",
                      expect=["PASS"])
    ns: dict[str, object] = {}
    exec(cell.source.split("\n")[2], ns)  # noqa: S102 - the generated field line only
    assert ns["TEXT"] == 'say "hi"\\now'
    lines = _lines(cell)
    assert lines[-2] == "print(TEXT)" and cell.expect == ("PASS",)


def test_form_default_with_line_separator_stays_one_line() -> None:
    cell = nbkit.form("f", "T", {"TEXT": "a b\nc"})
    line = cell.source.splitlines()[2]
    ns: dict[str, object] = {}
    exec(line, ns)  # noqa: S102 - the generated field line only
    assert ns["TEXT"] == "a b\nc"


def test_bad_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_min"):
        nbkit.code("a", "T", "pass", timeout_min=0)


@pytest.mark.parametrize("fields",[{}, {"bad name": "x"}, {"class": "x"}, {"X": 1.5},
                                    {"X": []}, {"X": ["a", 1]}])
def test_bad_form_fields_rejected(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        nbkit.form("f", "T", fields)


def test_notebook_validation_and_lookup() -> None:
    a = nbkit.code("a", "A", "pass")
    f = nbkit.form("f", "F", {"X": "y"})
    nb = nbkit.Notebook("C1N", 1, "env-check", "Env check", "chunk/C1N-notebook-pipeline",
                        [nbkit.md("hi"), a, f])
    assert nb.cells[1] is a and nb.gpu is True
    assert nb.cell("f") is f and nb.forms == (f,)
    assert [c.id for c in nb.checked_cells] == ["a", "f"]
    with pytest.raises(KeyError):
        nb.cell("missing")
    with pytest.raises(ValueError, match="duplicate"):
        nbkit.Notebook("C1N", 1, "s", "T", "b", [a, a])
    with pytest.raises(ValueError):
        nbkit.Notebook("C1N", 0, "s", "T", "b", [a])
    with pytest.raises(ValueError):
        nbkit.Notebook("C1N", 1, "bad slug", "T", "b", [a])
    with pytest.raises(ValueError):
        nbkit.Notebook("C1N", 1, "s", "T", "", [a])
