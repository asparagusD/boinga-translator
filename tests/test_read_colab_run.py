"""Tests for tools/read_colab_run.py: cell statuses, forms, binary stripping, trimming and the CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

import build_notebook
import nbformat
import pytest
import read_colab_run
from nbformat.v4 import new_output

SPEC = '''"""Reader test spec."""
from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C9", run=1, slug="read", title="Reader test", branch="chunk/C9-read",
    cells=[
        md("About it."),
        code("good", "Good cell", "print('PASS: good')", expect=["^PASS: good$"]),
        md("About it."),
        code("bad", "Bad cell", "print('PASS: bad')", expect=["^PASS: bad$"]),
        md("About it."),
        code("boom", "Boom cell", "1/0"),
        md("About it."),
        code("later", "Never ran", "pass"),
        md("About it."),
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


# ---- C1.8b: CLI, spec lookup, staleness, digest, --save-reports, exit code ----

HEADER = ("RUN-HEADER chunk=C9 run=1 branch=chunk/C9-read commit=abc1234 gpu=Tesla_T4 "
          "utc=2026-11-07T10:00:00+00:00\n")


@pytest.fixture
def cli_run(run: tuple[Path, read_colab_run.Spec], tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> Path:
    """The executed notebook with an HDR header; repo = tmp_path; git answers faked."""
    path, _spec = run
    _rewrite(path, "HDR", [{"output_type": "stream", "name": "stdout", "text": HEADER}])
    monkeypatch.setattr(read_colab_run.paths, "REPO", tmp_path)
    monkeypatch.setattr(read_colab_run, "_git", lambda *args: {
        ("rev-parse", "--short", "HEAD"): "def5678",
        ("log", "-1", "--format=%h", "--", "notebooks/runs/C9_read_run1.ipynb"): "abc1234",
    }.get(args, "") if args[0] != "merge-base" else "yes")
    return path


def test_cli_digest_save_reports_and_exit_code(
        cli_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert read_colab_run.main([str(cli_run), "--save-reports"]) == 1
    out = capsys.readouterr().out
    assert re.search(r"^\| bad \| Bad cell \| FAIL \|", out, re.MULTILINE)
    assert re.search(r"^\| good \| Good cell \| PASS \|", out, re.MULTILINE)
    digest = (tmp_path / "reports" / "colab_runs" / "C9_read_run1.md").read_text(encoding="utf-8")
    assert "commit `abc1234`" in digest and "GPU `Tesla_T4`" in digest
    assert "| obs | Listening notes | NOT RUN |" in digest
    assert '- `VOICE` = "male"' in digest and "- `OK` = False" in digest
    assert "### boom — Boom cell" in digest and "ZeroDivisionError" in digest
    assert B64[:40] not in digest and "ghp_" not in digest
    assert (tmp_path / "reports" / "x.json").read_text(encoding="utf-8") == "{}\n"
    assert "saved reports/x.json" in out


def test_spec_found_from_run_header_or_given(cli_run: Path, tmp_path: Path) -> None:
    nb = read_colab_run.load_executed(cli_run)
    assert read_colab_run.find_spec(nb).path == "notebooks/specs/read.py"
    nb_no_hdr = json.loads(cli_run.read_text(encoding="utf-8"))
    for c in nb_no_hdr["cells"]:
        c["outputs"] = []
    del nb_no_hdr["metadata"]["boinga"]
    cli_run.write_text(json.dumps(nb_no_hdr), encoding="utf-8")
    with pytest.raises(ValueError, match="--spec"):
        read_colab_run.find_spec(read_colab_run.load_executed(cli_run))
    spec_arg = str(tmp_path / "notebooks" / "specs" / "read.py")
    assert read_colab_run.main([str(cli_run), "--spec", spec_arg]) == 1


def test_staleness(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {"head": "def5678", "built": "abc1234", "ancestor": "yes"}

    def fake_git(*args: str) -> str:
        if args[0] == "merge-base":
            return answers["ancestor"]
        return answers["head"] if args[0] == "rev-parse" else answers["built"]

    monkeypatch.setattr(read_colab_run, "_git", fake_git)
    nb_rel = "notebooks/runs/C9_read_run1.ipynb"
    assert read_colab_run.staleness("abc1234", nb_rel).startswith("OK")
    assert "HEAD is def5678" in read_colab_run.staleness("abc1234", nb_rel)
    answers["ancestor"] = ""
    assert read_colab_run.staleness("0000000", nb_rel).startswith("STALE")
    assert read_colab_run.staleness("unknown", nb_rel).startswith("UNKNOWN")
    answers["built"] = "abc1234"
    real = fake_git
    monkeypatch.setattr(read_colab_run, "_git",
                        lambda *a: "" if a[0] == "cat-file" else real(*a))
    assert "not in this clone" in read_colab_run.staleness("9999999", nb_rel)


def test_executed_without_output_has_reason(run: tuple[Path, read_colab_run.Spec]) -> None:
    path, _spec = run
    _rewrite(path, "later", [])
    assert _results(run)["later"].reasons == ["executed but printed nothing"]


def test_exit_code_cells_and_staleness() -> None:
    ok = read_colab_run.CellResult("a", "A", "PASS", [], "", "", None)
    bad = read_colab_run.CellResult("b", "B", "NOT RUN", [], "", "", None)
    assert read_colab_run.exit_code([ok]) == 0
    assert read_colab_run.exit_code([ok, bad]) == 1
    assert read_colab_run.exit_code([ok], "STALE: the run used 0000000") == 1
    assert read_colab_run.exit_code([ok], "UNKNOWN: no RUN-HEADER") == 0
    assert read_colab_run.exit_code([ok], "OK: the run used abc1234") == 0


def test_spec_from_metadata_when_header_missing(cli_run: Path, tmp_path: Path) -> None:
    nb = json.loads(cli_run.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        c["outputs"] = []
    cli_run.write_text(json.dumps(nb), encoding="utf-8")
    spec = read_colab_run.find_spec(read_colab_run.load_executed(cli_run))
    assert spec.path == "notebooks/specs/read.py"
    (tmp_path / "notebooks" / "specs" / "_helper.py").write_text("raise SystemExit(9)\n",
                                                                encoding="utf-8")
    other = HEADER.replace("run=1", "run=7")
    _rewrite(cli_run, "HDR", [{"output_type": "stream", "name": "stdout", "text": other}])
    with pytest.raises(ValueError, match="no spec for chunk=C9 run=7"):
        read_colab_run.find_spec(read_colab_run.load_executed(cli_run))


def test_unsafe_block_path_is_an_error_not_a_traceback(
        cli_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = "=== BEGIN src/evil.py ===\nx\n=== END src/evil.py ===\n"
    _rewrite(cli_run, "bad", [{"output_type": "stream", "name": "stdout", "text": bad}])
    assert read_colab_run.main([str(cli_run), "--save-reports"]) == 1
    assert "no report files written" in capsys.readouterr().err
    assert not (tmp_path / "src").exists() and not (tmp_path / "reports" / "x.json").exists()


def test_cli_exits_1_on_stale_run(cli_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(read_colab_run, "exit_code",
                        lambda results, stale="": 1 if stale.startswith("STALE") else 0)
    monkeypatch.setattr(read_colab_run, "staleness", lambda commit, rel: "STALE: old run")
    assert read_colab_run.main([str(cli_run)]) == 1
    monkeypatch.setattr(read_colab_run, "staleness", lambda commit, rel: "OK: fine")
    assert read_colab_run.main([str(cli_run)]) == 0


def test_two_specs_with_same_chunk_and_run(cli_run: Path, tmp_path: Path) -> None:
    specs = tmp_path / "notebooks" / "specs"
    (specs / "a_boot.py").write_text(SPEC.replace('slug="read"', 'slug="boot"'), encoding="utf-8")
    nb = read_colab_run.load_executed(cli_run)
    assert read_colab_run.find_spec(nb).path == "notebooks/specs/read.py"  # metadata decides
    del nb.metadata["boinga"]
    with pytest.raises(ValueError, match="matches"):
        read_colab_run.find_spec(nb)
    found = read_colab_run.find_spec(nb, "C9_read_run1 (1).ipynb")  # browser-renamed download
    assert found.path == "notebooks/specs/read.py"
