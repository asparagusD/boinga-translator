"""Tests for colab_runtime.py: run header, cell marker, notes, expect_files and play fallbacks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from boinga_translator import colab_runtime, paths


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """REPO, ROOT and FAST under tmp_path; local mode; no torch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError
    return repo


def test_run_header_without_git_or_torch(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_git(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    colab_runtime.run_header("C1N", 3)
    out = capsys.readouterr().out
    assert re.search(r"^RUN-HEADER chunk=C1N run=3 branch=unknown commit=unknown gpu=none "
                     r"utc=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$", out, re.MULTILINE)
    assert "WARN: no GPU" in out
    assert "=== BEGIN reports/env.json ===" in out and "PASS: env recorded" in out
    assert not re.search(r"^FAIL:", out, re.MULTILINE)
    assert (sandbox / "reports" / "env.json").is_file()


def test_run_header_reports_branch_and_gpu(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_git(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        out = "chunk/C1N-x\n" if "--abbrev-ref" in cmd else "abc1234\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(subprocess, "run", fake_git)
    monkeypatch.setattr(colab_runtime.env_report, "_torch_info", lambda: {
        "torch": "2.0", "torch_cuda": "12", "torch_error": None,
        "gpu_name": "Tesla T4", "gpu_mem_gb": 14.6})
    colab_runtime.run_header("C2", 1)
    out = capsys.readouterr().out
    assert "RUN-HEADER chunk=C2 run=1 branch=chunk/C1N-x commit=abc1234 gpu=Tesla_T4 " in out
    assert "WARN: no GPU" not in out


def test_cell_ok(capsys: pytest.CaptureFixture[str]) -> None:
    colab_runtime.cell_ok("baseline.1")
    assert capsys.readouterr().out == "CELL-OK baseline.1\n"


def test_record_notes_writes_emits_and_flags_empty(
    sandbox: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = {"VOICE": "female", "NOTES": "  ", "SCORE": 0, "CLEAR": False, "WHY": ""}
    repo_path = colab_runtime.record_notes("obs", values)
    out = capsys.readouterr().out
    assert repo_path == sandbox / "reports" / "notes" / "obs.json"
    assert json.loads(repo_path.read_text(encoding="utf-8")) == values
    assert (tmp_path / "root" / "reports" / "reports" / "notes" / "obs.json").is_file()
    assert "=== BEGIN reports/notes/obs.json ===" in out
    missing = re.findall(r"^NOTES-MISSING (\S+)$", out, re.MULTILINE)
    assert missing == ["NOTES", "WHY"]  # only empty strings; 0 / False are answers


def test_record_notes_keeps_bangla(sandbox: Path) -> None:
    path = colab_runtime.record_notes("obs", {"TEXT": "আমি"})
    assert "আমি" in path.read_text(encoding="utf-8")


def test_record_notes_rejects_unsafe_id(sandbox: Path) -> None:
    for bad in ("../escape", "", "  "):
        with pytest.raises(ValueError):
            colab_runtime.record_notes(bad, {"X": "y"})


def test_expect_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good, small = tmp_path / "a.wav", tmp_path / "b.wav"
    good.write_bytes(b"x" * 10)
    small.write_bytes(b"x")
    assert colab_runtime.expect_files([good], min_bytes=5) is True
    assert colab_runtime.expect_files([good, small, tmp_path / "none.wav"], min_bytes=5,
                                      stop=False) is False
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"PASS: {good} (10 bytes)"
    assert lines[2] == f"FAIL: {small} too small (1 < 5 bytes)"
    assert lines[3] == f"FAIL: {tmp_path / 'none.wav'} missing"


def test_play_without_ipython_prints_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)
    colab_runtime.play([tmp_path / "a.wav", tmp_path / "b.wav"], ["female 1", "male 1"])
    assert capsys.readouterr().out.splitlines() == [
        f"AUDIO female 1: {tmp_path / 'a.wav'}", f"AUDIO male 1: {tmp_path / 'b.wav'}"]
    with pytest.raises(ValueError, match="labels"):
        colab_runtime.play([tmp_path / "a.wav"], [])


def test_play_in_ipython_displays_players(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    shown: list[Any] = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: object()  # type: ignore[attr-defined]
    display_mod = types.ModuleType("IPython.display")
    display_mod.Audio = lambda filename: ("Audio", filename)  # type: ignore[attr-defined]
    display_mod.display = shown.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    colab_runtime.play([wav, tmp_path / "gone.wav"], ["female 1", "male 1"])
    out = capsys.readouterr().out.splitlines()
    assert shown == [("Audio", str(wav))]
    assert out == ["AUDIO female 1:", f"WARN: missing audio for male 1: {tmp_path / 'gone.wav'}"]


def test_expect_files_stops_the_run_by_default(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        colab_runtime.expect_files([tmp_path / "none.wav"])
    assert exc.value.code == 1
    assert capsys.readouterr().out == f"FAIL: {tmp_path / 'none.wav'} missing\n"


def test_fail_prints_and_stops(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        colab_runtime.fail("checkpoint missing")
    assert exc.value.code == 1
    assert capsys.readouterr().out == "FAIL: checkpoint missing\n"
