"""Tests for run_check.py and `boinga record-run` / `check-run` (fake run folder on fake Drive)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boinga_translator import cli, milestones, paths, run_check


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """run1 folder with checkpoints 10000/20000 and a config on a fake Drive."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "vm"))
    folder = paths.run_dir("run1") / "run1-November-08-2026_10+00AM-abc1234"
    folder.mkdir(parents=True)
    for step in (10000, 20000):
        (folder / f"checkpoint_{step}.pth").write_bytes(b"w" * step)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    return folder


def test_record_on_the_vm(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_check.record("run1") == 3
    data = json.loads(run_check.manifest_path("run1").read_text(encoding="utf-8"))
    assert run_check.manifest_path("run1").is_relative_to(paths.fast())  # never on Drive
    assert list(data["files"]) == ["checkpoint_10000.pth", "checkpoint_20000.pth",
                                   "config.json"]
    assert data["files"]["checkpoint_20000.pth"] == {
        "bytes": 20000, "md5": milestones.md5(run / "checkpoint_20000.pth")}
    assert "PASS: recorded 3 files of run run1" in capsys.readouterr().out


def test_check_through_the_api(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_check.record("run1")
    want = json.loads(run_check.manifest_path("run1").read_text(encoding="utf-8"))["files"]
    good = {n: {"size": w["bytes"], "md5": w["md5"]} for n, w in want.items()}
    assert run_check.check("run1", lambda folder: good) == 3
    assert "PASS: 3 files of run run1 on Drive (Drive API)" in capsys.readouterr().out
    lost = {k: v for k, v in good.items() if k != "checkpoint_20000.pth"}  # upload failed
    short = {**good, "checkpoint_10000.pth": {"size": 5, "md5": "x"}}
    other = {**good, "checkpoint_10000.pth": {"size": 10000, "md5": "x"}}
    for remote, why in ((lost, "MISSING"), (short, "5 bytes"), (other, "md5 differs")):
        with pytest.raises(RuntimeError, match="not on Drive"):
            run_check.check("run1", lambda folder, r=remote: r)
        assert why in capsys.readouterr().out


def test_check_whole_folder_missing(run: Path) -> None:
    run_check.record("run1")

    def remote(folder: Path) -> dict:
        raise FileNotFoundError(f"{folder} is not on Drive")

    with pytest.raises(RuntimeError, match="run folder not on Drive"):
        run_check.check("run1", remote)


def test_check_needs_a_record(run: Path) -> None:
    with pytest.raises(FileNotFoundError, match="record-run"):
        run_check.check("run1")


def test_cli(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check-run", "--run", "run1"]) == 1
    assert "FAIL: check-run FileNotFoundError" in capsys.readouterr().out
    assert cli.main(["record-run", "--run", "run1"]) == 0
    assert cli.main(["check-run", "--run", "run1"]) == 0  # the mount view: sizes only
    assert "(mount)" in capsys.readouterr().out
    assert cli.main(["record-run", "--run", "nope"]) == 1
