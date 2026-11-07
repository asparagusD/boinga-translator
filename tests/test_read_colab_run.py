"""Tests for tools/read_colab_run.py core: statuses, form values, binary stripping, trimming."""

from __future__ import annotations

import json
from pathlib import Path

import build_notebook
import nbformat
import pytest
import read_colab_run
from nbformat.v4 import new_output

SPEC = '''"""Reader test spec."""
from nbkit import Notebook, code, form

NOTEBOOK = Notebook(
    chunk="C9", run=1, slug="read", title="Reader test", branch="chunk/C9-read",
    cells=[
        code("good", "Good cell", "print('PASS: good')", expect=["^PASS: good$"]),
        code("bad", "Bad cell", "print('PASS: bad')", expect=["^PASS: bad$"]),
        code("boom", "Boom cell", "1/0"),
        code("later", "Never ran", "pass"),
        form("obs", "Listening notes", {"VOICE": ["female", "male"], "OK": True, "WHY": "x"}),
    ],
)
'''
B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=" * 40


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, read_colab_run.Spec]:
    """A built notebook from SPEC, then 'executed' by adding outputs as Colab would."""
    (tmp_path / "notebooks" / "specs").mkdir(parents=True)
    real = build_notebook.ROOT / "notebooks" / "colab_bootstrap.py"
    (tmp_path / "notebooks" / "colab_bootstrap.py").write_text(
        real.read_text(encoding="utf-8"), encoding="utf-8")
    spec_path = tmp_path / "notebooks" / "specs" / "read.py"
    spec_path.write_text(SPEC, encoding="utf-8")
    monkeypatch.setattr(build_notebook, "ROOT", tmp_path)
    _rel, text = build_notebook.build(spec_path)
    nb = nbformat.reads(text, as_version=4)
    by_id = {c.metadata.get("boinga", {}).get("id"): c for c in nb.cells}
    del by_id["good"].metadata["boinga"]  # Colab may drop metadata: ids come from the source

    def ran(cid: str, *outputs: dict) -> None:
        by_id[cid].execution_count = 1
        by_id[cid].outputs = list(outputs)

    ran("good", new_output("stream", name="stdout", text=(
            f"PASS: good\nprinted data:audio/wav;base64,{B64}\n"
            "=== BEGIN reports/x.json ===\n{}\n=== END reports/x.json ===\n"
            "token ghp_abcdefghijklmnopqrstuvwxyz0123456789\n")),
        new_output("display_data", data={
            "text/plain": "<IPython.lib.display.Audio object>",
            "text/html": f'<audio src="data:audio/wav;base64,{B64}"></audio>',
            "audio/wav": B64, "image/png": B64}),
        new_output("stream", name="stdout", text="CELL-OK good\n"))
    ran("bad", new_output("stream", name="stdout", text="FAIL: nope\nCELL-OK bad\n"))
    ran("boom", new_output("error", ename="ZeroDivisionError", evalue="division by zero",
                           traceback=["\x1b[0;31mTraceback (most recent call last)\x1b[0m",
                                      *[f"  frame {i}" for i in range(40)],
                                      "\x1b[0;31mZeroDivisionError\x1b[0m: division by zero"]))
    src = "".join(by_id["obs"].source).replace('VOICE = "female"', 'VOICE = "male"')
    by_id["obs"].source = src.replace("OK = True", "OK = False")
    out = tmp_path / "executed.ipynb"
    nbformat.write(nb, out)
    return out, read_colab_run.load_spec(spec_path)


def _results(run: tuple[Path, read_colab_run.Spec]) -> dict[str, read_colab_run.CellResult]:
    path, spec = run
    return {r.id: r for r in read_colab_run.check_run(read_colab_run.load_executed(path), spec)}


def test_statuses(run: tuple[Path, read_colab_run.Spec]) -> None:
    res = _results(run)
    assert res["good"].status == "PASS", res["good"].reasons
    assert res["bad"].status == "FAIL"
    assert any("^PASS: bad$" in r for r in res["bad"].reasons)
    assert any("^FAIL:" in r for r in res["bad"].reasons)
    assert res["boom"].status == "ERROR"
    assert "ZeroDivisionError: division by zero" in res["boom"].reasons[0]
    assert res["later"].status == "NOT RUN" and res["obs"].status == "NOT RUN"
    assert res["BOOT.1"].status == "NOT RUN" and res["HDR"].status == "NOT RUN"
    assert list(res) == ["BOOT.1", "BOOT.2", "BOOT.3", "BOOT.4", "HDR", "good", "bad", "boom",
                         "later", "obs"]


def test_error_keeps_last_30_traceback_lines_without_ansi(
        run: tuple[Path, read_colab_run.Spec]) -> None:
    text = _results(run)["boom"].text
    lines = text.splitlines()
    assert "\x1b" not in text
    assert lines[-1] == "ZeroDivisionError: division by zero"
    assert len([ln for ln in lines if ln.startswith("  frame")]) == 29


def test_form_values_from_edited_source(run: tuple[Path, read_colab_run.Spec]) -> None:
    assert _results(run)["obs"].form_values == {"VOICE": "male", "OK": False, "WHY": "x"}
    assert _results(run)["good"].form_values is None


def test_binary_dropped_and_tokens_redacted(run: tuple[Path, read_colab_run.Spec]) -> None:
    text = _results(run)["good"].text
    assert B64[:40] not in text and "<audio" not in text
    assert "printed data:audio/wav;base64,[base64 dropped]" in text
    assert "<IPython.lib.display.Audio object>" in text
    assert "ghp_" not in text and "***" in text
    assert "=== BEGIN reports/x.json ===" in text


def test_raw_text_keeps_blocks_for_saving(run: tuple[Path, read_colab_run.Spec]) -> None:
    raw = _results(run)["good"].raw
    assert "=== BEGIN reports/x.json ===\n{}\n=== END reports/x.json ===" in raw
    assert B64 in raw and "ghp_" not in raw  # blocks are saved verbatim, tokens never


def _rewrite(path: Path, cid: str, outputs: list[dict]) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if "".join(c["source"]).startswith(f"# [boinga {cid}] "):
            c["execution_count"], c["outputs"] = 1, outputs
    path.write_text(json.dumps(nb), encoding="utf-8")


def test_cell_ok_required_only_for_cells_that_call_it(
        run: tuple[Path, read_colab_run.Spec]) -> None:
    path, _spec = run
    stdout = {"output_type": "stream", "name": "stdout"}
    _rewrite(path, "later", [{**stdout, "text": "done\n"}])
    _rewrite(path, "HDR", [{**stdout, "text": "RUN-HEADER chunk=C9 run=1\n"}])
    res = _results(run)
    assert res["later"].status == "FAIL"
    assert res["later"].reasons == ["no CELL-OK line (cell did not reach its end)"]
    assert res["HDR"].status == "PASS"


def test_duplicate_cell_last_copy_wins(run: tuple[Path, read_colab_run.Spec]) -> None:
    path, _spec = run
    nb = json.loads(path.read_text(encoding="utf-8"))
    copy = json.loads(json.dumps(next(
        c for c in nb["cells"] if "".join(c["source"]).startswith("# [boinga bad] "))))
    copy["id"] = "bad-copy"
    copy["outputs"] = [{"output_type": "stream", "name": "stdout",
                        "text": "PASS: bad\nCELL-OK bad\n"}]
    nb["cells"].append(copy)
    path.write_text(json.dumps(nb), encoding="utf-8")
    assert _results(run)["bad"].status == "PASS"


def test_form_value_fallback_and_redaction() -> None:
    src = ('A = foo bar #@param {type:"raw"}\n'
           'B = "ghp_abcdefghij0123456789" #@param {type:"string"}')
    assert read_colab_run.form_values(src) == {"A": "foo bar", "B": "***"}


def test_missing_cell_reported(run: tuple[Path, read_colab_run.Spec]) -> None:
    path, _spec = run
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["cells"] = [c for c in nb["cells"] if "[boinga later]" not in "".join(c["source"])]
    path.write_text(json.dumps(nb), encoding="utf-8")
    later = _results(run)["later"]
    assert later.status == "NOT RUN" and later.reasons == ["cell not found in the notebook"]


def test_trim_keeps_head_tail_and_marker_lines() -> None:
    lines = [f"line {i}" for i in range(300)]
    lines[100], lines[150], lines[200] = "PASS: mid", "=== BEGIN reports/a.json ===", "WARN: w"
    out = read_colab_run.trim("\n".join(lines)).splitlines()
    assert out[:60] == lines[:60] and out[-60:] == lines[-60:]
    assert "PASS: mid" in out and "=== BEGIN reports/a.json ===" in out and "WARN: w" in out
    assert any("lines trimmed" in ln for ln in out)
    assert read_colab_run.trim("a\nb") == "a\nb"


def test_digest_keeps_banner_lines() -> None:
    banner = "=" * 120
    assert read_colab_run.digest_text(banner) == banner
