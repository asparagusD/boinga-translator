"""Tests for milestones.py and `boinga keep` (fake run folder, pickle instead of torch)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from boinga_translator import ckpt_sampler, cli, milestones, paths

LOG = "".join(
    f"\n   --> TIME: x -- GLOBAL_STEP: {s}\n\n   --> EVAL PERFORMANCE\n"
    f"     | > avg_loss_0: 2.0\n     | > avg_loss_1: 40.0\n     | > avg_loss_mel: {mel}\n"
    for s, mel in ((375, 25.0), (775, 22.0), (1175, 23.0), (1575, 24.0), (1975, 26.0)))


def _state(step: int) -> dict[str, Any]:
    return {"model": {"w": [step, 1.0], "b": [0.5]}, "optimizer": {"big": [0] * 1000},
            "step": step, "epoch": 1}


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run folder with checkpoints 400..2000 (pickled fake states), a config and a log."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    folder = paths.run_dir("run1") / "run1-November-08-2026_10+00AM-abc1234"
    folder.mkdir(parents=True)
    for step in (400, 800, 1200, 1600, 2000):
        (folder / f"checkpoint_{step}.pth").write_bytes(pickle.dumps(_state(step)))
    (folder / "config.json").write_text('{"run_name": "run1"}', encoding="utf-8")
    (folder / ckpt_sampler.LOG_FILE).write_text(LOG, encoding="utf-8")
    monkeypatch.setattr(milestones, "_load", lambda p: pickle.loads(Path(p).read_bytes()))
    monkeypatch.setattr(milestones, "_save", lambda obj, p: Path(p).write_bytes(pickle.dumps(obj)))
    return folder


def test_default_steps_are_the_listened_ones(run: Path) -> None:
    # 4 best by eval loss_mel (800, 1200, 1600, 400) + the last (2000)
    assert milestones.default_steps("run1") == [400, 800, 1200, 1600, 2000]


def test_keep_writes_model_only_copies(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert milestones.keep("run1", [800, 2000]) == 2
    out = milestones.milestone_dir("run1", 800)
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "keep.json", "model.pth"]
    kept = pickle.loads((out / "model.pth").read_bytes())
    assert kept == {"model": _state(800)["model"], "step": 800, "epoch": 1}  # no optimizer
    assert (out / "config.json").read_text(encoding="utf-8") == '{"run_name": "run1"}'
    meta = json.loads((out / "keep.json").read_text(encoding="utf-8"))
    assert meta["source"] == "checkpoint_800.pth" and meta["n_tensors"] == 2
    assert meta["source_folder"] == run.name
    assert meta["bytes"] == (out / "model.pth").stat().st_size
    text = capsys.readouterr().out
    assert "step 800: kept" in text and "PASS: kept 2 milestones (0 already kept)" in text
    assert milestones.keep("run1", [800, 2000]) == 0  # idempotent
    assert "PASS: kept 0 milestones (2 already kept)" in capsys.readouterr().out


def test_keep_refuses_a_missing_step(run: Path) -> None:
    with pytest.raises(FileNotFoundError, match="step 999"):
        milestones.keep("run1", [800, 999])
    assert not milestones.milestone_dir("run1", 800).exists()  # nothing written


def test_keep_detects_a_bad_copy(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = milestones._load
    monkeypatch.setattr(milestones, "_load", lambda p: (
        {"model": {"w": [0, 0.0], "b": [0.5]}} if Path(p).name.startswith("model.pth") else real(p)))
    with pytest.raises(RuntimeError, match="differs from"):
        milestones.keep("run1", [800])
    out = milestones.milestone_dir("run1", 800)
    assert not (out / "keep.json").exists() and not (out / "model.pth").exists()


def test_keep_resumes_an_interrupted_copy(run: Path) -> None:
    out = milestones.milestone_dir("run1", 800)
    out.mkdir(parents=True)
    (out / "model.pth.part").write_bytes(b"half")  # Drive dropped mid-copy
    (out / "keep.json.part").write_text("{", encoding="utf-8")
    assert milestones.keep("run1", [800]) == 1
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "keep.json", "model.pth"]


def test_keep_without_evals_keeps_the_last(run: Path) -> None:
    (run / ckpt_sampler.LOG_FILE).write_text("", encoding="utf-8")
    assert milestones.default_steps("run1") == [2000]  # never an empty default


def test_cli(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["keep", "--run", "run1", "--steps", "1200"]) == 0
    assert milestones.milestone_dir("run1", 1200).joinpath("keep.json").is_file()
    assert cli.main(["keep", "--run", "run1"]) == 0  # default: the listened steps
    assert "PASS: kept 4 milestones (1 already kept)" in capsys.readouterr().out
    assert cli.main(["keep", "--run", "nope"]) == 1
    assert "FAIL: keep FileNotFoundError" in capsys.readouterr().out


def test_keep_needs_drive_space(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(milestones.train_runner, "free_gb", lambda path: 0.0)
    with pytest.raises(RuntimeError, match="free on Drive"):
        milestones.keep("run1", [800])
    assert not milestones.milestone_dir("run1", 800).exists()


def test_check(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(FileNotFoundError, match="boinga keep"):
        milestones.check("run1")
    milestones.keep("run1", [800, 2000])
    assert milestones.check("run1") == 2
    assert "PASS: 2 milestones on Drive" in capsys.readouterr().out
    (milestones.milestone_dir("run1", 2000) / "model.pth").write_bytes(b"short")  # lost upload
    with pytest.raises(RuntimeError, match="step_2000"):
        milestones.check("run1")
    assert cli.main(["check-milestones", "--run", "run1"]) == 1
    assert "FAIL: check-milestones RuntimeError" in capsys.readouterr().out


def test_keep_refuses_another_runs_milestone(run: Path) -> None:
    milestones.keep("run1", [800])
    run.rename(run.with_name("run1-November-09-2026_09+00AM-def5678"))  # run1 retrained
    with pytest.raises(RuntimeError, match="move it away"):
        milestones.keep("run1", [800])
