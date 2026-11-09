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
    assert cell.allow_interrupt is False  # default: no metadata key, old notebooks unchanged


def test_code_allow_interrupt() -> None:
    cell = nbkit.code("live", "Live demo", "serve()", allow_interrupt=True)
    assert cell.allow_interrupt is True and cell.metadata["boinga"]["allow_interrupt"] is True
    assert cell.source.endswith(CELL_OK.format("live"))


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


def test_code_skip_unless_wraps_src() -> None:
    cell = nbkit.code("S10", "Smoke", "x = 1\nprint('PASS: smoke')", expect=["^PASS: smoke"],
                      skip_unless="RUN_TRAINING")
    assert _lines(cell)[1:6] == ["if not RUN_TRAINING:",
                                 "    print('SKIPPED (RUN_TRAINING=False)')", "else:",
                                 "    x = 1", "    print('PASS: smoke')"]
    assert cell.source.endswith(CELL_OK.format("S10"))  # cell_ok runs on both branches
    assert cell.skip_unless == "RUN_TRAINING"
    assert cell.metadata["boinga"]["skip_unless"] == "RUN_TRAINING"
    assert nbkit.skip_line("RUN_TRAINING") == "SKIPPED (RUN_TRAINING=False)"
    body = compile(cell.source.rsplit("\n", 1)[0], "<cell>", "exec")
    for flag, x in ((False, None), (True, 1)):
        ns: dict[str, object] = {"RUN_TRAINING": flag}
        exec(body, ns)  # noqa: S102
        assert ns.get("x") == x


def test_code_skip_unless_keeps_shell_lines() -> None:
    cell = nbkit.code("S11", "Train", "!boinga train --run run1", expect=["STOP"],
                      skip_unless="RUN_TRAINING")
    assert "else:\n    !boinga train --run run1\n" in cell.source


@pytest.mark.parametrize("src", ['s = """a\nb"""', "s = 'a\\\nb'", 'print(f"""{1}\nx""")',
                                 's = """never closed'])
def test_code_skip_unless_refuses_multiline_strings(src: str) -> None:
    with pytest.raises(ValueError, match="multi-line string"):
        nbkit.code("a", "A", src, skip_unless="RUN_TRAINING")
    nbkit.code("a", "A", src.replace("\n", " ").replace("\\", ""))  # fine without a flag


def test_code_skip_unless_validation() -> None:
    assert "skip_unless" not in nbkit.code("a", "A", "pass").metadata["boinga"]
    for bad in ("RUN TRAINING", "1flag", "if"):
        with pytest.raises(ValueError, match="identifier"):
            nbkit.code("a", "A", "pass", skip_unless=bad)
    with pytest.raises(ValueError, match="non-empty"):
        nbkit.code("a", "A", "", skip_unless="RUN_TRAINING")
    with pytest.raises(ValueError, match="shell"):  # still needs expect around a ! line
        nbkit.code("a", "A", "!boinga train --run run1", skip_unless="RUN_TRAINING")


@pytest.mark.parametrize("output", ["notebooks/boinga_translator.ipynb",
                                    "notebooks/runs/x_y.ipynb", None])
def test_notebook_output_ok(output: str | None) -> None:
    nb = nbkit.Notebook(chunk="C12", run=1, slug="final", title="T", branch="b", cells=[],
                        output=output)
    assert nb.output == output


@pytest.mark.parametrize("output", ["boinga.ipynb", "notebooks/../x.ipynb", "notebooks/x.py",
                                    "notebooks\\x.ipynb", "/notebooks/x.ipynb",
                                    "notebooks/specs/x.ipynb", "notebooks/a b.ipynb"])
def test_notebook_output_rejected(output: str) -> None:
    with pytest.raises(ValueError, match="output"):
        nbkit.Notebook(chunk="C12", run=1, slug="final", title="T", branch="b", cells=[],
                       output=output)
