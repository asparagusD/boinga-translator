"""Tests for train_runner.run() and `boinga train`: TrainerArgs, step stop, boinga_run.json (fakes)."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from boinga_translator import cli, paths, train_config, train_runner


class FakeTrainer:
    """Steps until a callback raises (like coqui's _fit), records saves."""

    instances: ClassVar[list[FakeTrainer]] = []
    total_epoch_steps = 100
    fail_with: BaseException | None = None

    def __init__(self, args: Any, config: Any, output_path: str | None = None, *,
                 model: Any = None, train_samples: Any = None, eval_samples: Any = None,
                 parse_command_line_args: bool = True, callbacks: dict | None = None) -> None:
        self.args, self.config, self.model = args, config, model
        self.train_samples, self.eval_samples = train_samples, eval_samples
        self.parse_command_line_args = parse_command_line_args
        self.callbacks = callbacks or {}
        self.total_steps_done = 0
        if getattr(args, "continue_path", ""):  # like coqui: same folder, step of last ckpt + 1
            self.output_path = Path(args.continue_path)
            steps = [s for p in self.output_path.glob("checkpoint_*.pth")
                     if (s := train_runner._ckpt_step(p)) is not None]
            self.total_steps_done = max(steps) + 1
        else:
            n = len(FakeTrainer.instances)  # coqui dates each folder: attempts never collide
            self.output_path = (Path(str(output_path))
                                / f"smoke-November-08-2026_10+0{n}AM-abc1234")
            self.output_path.mkdir(parents=True)
        self.saved: list[int] = []
        self.finished = False
        self.dashboard_logger = types.SimpleNamespace(finish=self._finish)
        FakeTrainer.instances.append(self)

    def _finish(self) -> None:
        self.finished = True

    def _fit(self) -> None:
        record = json.loads((self.output_path.parent / train_runner.RUN_RECORD).read_text("utf-8"))
        self.record_at_start = record["sessions"][-1]  # what run() wrote before training
        if FakeTrainer.fail_with is not None:
            self.total_steps_done = 7
            raise FakeTrainer.fail_with
        for _ in range(FakeTrainer.total_epoch_steps):
            self.total_steps_done += 1
            if "on_train_step_end" in self.callbacks:
                self.callbacks["on_train_step_end"](self)

    def save_checkpoint(self) -> None:
        self.saved.append(self.total_steps_done)
        (self.output_path / f"checkpoint_{self.total_steps_done}.pth").write_bytes(b"x")

    def fit(self) -> None:  # pragma: no cover - run() must not call it (os._exit on interrupt)
        raise AssertionError("run() must call _fit(), not fit()")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake coqui (config, model, samples, Trainer), a fake build() and fake torch memory."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    voice = paths.pretrained() / "male"
    voice.mkdir(parents=True)
    (voice / "model.pth").write_bytes(b"pth")
    FakeTrainer.instances, FakeTrainer.fail_with = [], None
    seen: dict[str, Any] = {}

    def build(run_name: str) -> dict:
        seen["built"] = run_name
        path = paths.REPO / train_config.CONFIG_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {}

    class VitsConfig:
        def load_json(self, path: str) -> None:
            seen["config_path"] = path

    def load_tts_samples(config: Any, *, eval_split: bool = True) -> tuple[list, list]:
        return [{"t": 1}] * 4, [{"t": 2}]

    coqui = types.SimpleNamespace(
        VitsConfig=VitsConfig, Vits=lambda config: ("model", config),
        load_tts_samples=load_tts_samples, Trainer=FakeTrainer,
        TrainerArgs=lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(train_config, "base_voice", lambda: "male")
    monkeypatch.setattr(train_config, "build", build)
    monkeypatch.setattr(train_runner, "_coqui", lambda: coqui)
    monkeypatch.setattr(train_runner, "_peak_memory_gb", lambda: 7.25)
    monkeypatch.setattr(train_runner.env_report, "collect",
                        lambda: {"git_commit": "abc1234", "gpu_name": "Tesla T4"})
    return seen


def _record(name: str = "smoke") -> dict:
    return json.loads((paths.run_dir(name) / train_runner.RUN_RECORD).read_text("utf-8"))


def test_new_run_restores_pretrained(fake: dict[str, Any]) -> None:
    session = train_runner.run("smoke")
    (tr,) = FakeTrainer.instances
    assert tr.args.restore_path == str(paths.pretrained() / "male" / "model.pth")
    assert not getattr(tr.args, "continue_path", "")
    assert tr.parse_command_line_args is False  # our CLI flags must not reach coqpit
    assert tr.output_path.parent == paths.run_dir("smoke")
    assert tr.model[0] == "model" and len(tr.train_samples) == 4 and len(tr.eval_samples) == 1
    assert tr.callbacks == {} and tr.saved == [] and tr.finished
    assert fake["built"] == "smoke"
    assert fake["config_path"] == str(paths.REPO / train_config.CONFIG_REL)
    assert session["steps_done"] == 100 and session["stopped_early"] is False


def test_max_steps_stops_and_saves(fake: dict[str, Any],
                                   capsys: pytest.CaptureFixture[str]) -> None:
    session = train_runner.run("smoke", max_steps=50)
    (tr,) = FakeTrainer.instances
    assert tr.total_steps_done == 50 and tr.saved == [50]
    assert "STOP at step 50" in capsys.readouterr().out
    assert session["stopped_early"] is True and session["steps_done"] == 50


def test_run_record(fake: dict[str, Any]) -> None:
    train_runner.run("smoke", max_steps=10)
    rec = _record()
    assert rec["run"] == "smoke"
    (s,) = rec["sessions"]
    assert s["git_commit"] == "abc1234" and s["env"]["gpu_name"] == "Tesla T4"
    assert s["config_path"] == str(paths.REPO / train_config.CONFIG_REL)
    assert s["restore_path"].endswith("model.pth") and s["max_steps"] == 10
    assert s["start_utc"] <= s["end_utc"]
    assert s["peak_gpu_memory_gb"] == 7.25 and s["steps_done"] == 10
    assert s["output_path"].startswith(str(paths.run_dir("smoke")))


def test_record_written_before_training(fake: dict[str, Any]) -> None:
    FakeTrainer.fail_with = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        train_runner.run("smoke", max_steps=10)
    (s,) = _record()["sessions"]
    assert s["error"] == "RuntimeError: CUDA out of memory"
    assert s["peak_gpu_memory_gb"] == 7.25 and "start_utc" in s


def _folder(name: str, *files: str) -> Path:
    """A fake coqui run folder under runs/smoke with the given files."""
    folder = paths.run_dir("smoke") / name
    folder.mkdir(parents=True)
    for f in files:
        (folder / f).write_bytes(b"x")
    return folder


def test_latest_run_folder_highest_step_not_mtime(fake: dict[str, Any]) -> None:
    assert train_runner.latest_run_folder("smoke") is None  # no run folder at all
    _folder("smoke-empty")
    _folder("smoke-best-only", "best_model_900.pth", "checkpoint_abc.pth", "config.json")
    assert train_runner.latest_run_folder("smoke") is None  # no checkpoint_*.pth anywhere
    high = _folder("smoke-a", "checkpoint_500.pth", "checkpoint_1000.pth", "config.json")
    newer = _folder("smoke-b", "checkpoint_100.pth", "config.json")
    os.utime(newer / "checkpoint_100.pth", (2_000_000_000, 2_000_000_000))  # newest mtime
    assert train_runner.latest_run_folder("smoke") == (high, 1000)


def test_resume_uses_continue_path_never_restore(fake: dict[str, Any],
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    folder = _folder("smoke-a", "checkpoint_50.pth", "config.json")
    session = train_runner.run("smoke", max_steps=60)
    (tr,) = FakeTrainer.instances
    assert tr.args.continue_path == str(folder)
    assert not hasattr(tr.args, "restore_path")  # never restore on a resume (skill)
    assert "built" not in fake  # the run keeps its own config.json, no rebuild
    assert fake["config_path"] == str(folder / "config.json")
    assert f"RESUME from step 50 in {folder}" in capsys.readouterr().out
    assert tr.total_steps_done == 60 and tr.saved == [60]  # max_steps counts total steps
    assert session["kind"] == "resume" and session["resumed_from_step"] == 50
    (s,) = _record()["sessions"]
    assert s["continue_path"] == str(folder) and "restore_path" not in s


def test_resume_without_max_steps(fake: dict[str, Any]) -> None:
    folder = _folder("smoke-a", "checkpoint_50.pth", "config.json")
    session = train_runner.run("smoke")  # no limit: resume and train to the end of _fit
    (tr,) = FakeTrainer.instances
    assert tr.args.continue_path == str(folder) and tr.callbacks == {}
    assert session["kind"] == "resume" and session["steps_done"] == 151


def test_run_name_coqui_would_misread(fake: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="checkpoint search"):
        train_runner.run("checkpoint_5")


def test_already_at_max_steps(fake: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    _folder("smoke-a", "checkpoint_60.pth", "config.json")
    session = train_runner.run("smoke", max_steps=60)
    assert "PASS: run smoke already at step 60" in capsys.readouterr().out
    assert FakeTrainer.instances == [] and session["kind"] == "done"
    assert not (paths.run_dir("smoke") / train_runner.RUN_RECORD).exists()


def test_resume_needs_the_run_config(fake: dict[str, Any]) -> None:
    _folder("smoke-a", "checkpoint_50.pth")
    with pytest.raises(FileNotFoundError, match="config.json"):
        train_runner.run("smoke", max_steps=60)


def test_missing_pretrained_model(fake: dict[str, Any]) -> None:
    (paths.pretrained() / "male" / "model.pth").unlink()
    with pytest.raises(FileNotFoundError, match="model.pth"):
        train_runner.run("smoke")


@pytest.mark.parametrize("bad", [0, -5])
def test_bad_max_steps(fake: dict[str, Any], bad: int) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        train_runner.run("smoke", max_steps=bad)


def test_cli_train(fake: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["train", "--run", "smoke", "--max-steps", "20"]) == 0
    assert "PASS: run smoke at step 20" in capsys.readouterr().out
    (FakeTrainer.instances[0].output_path / "config.json").write_text("{}", encoding="utf-8")
    assert cli.main(["train", "--run", "smoke", "--max-steps", "20"]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS:") == 1 and "PASS: run smoke already at step 20" in out
    assert cli.main(["train", "--run", "smoke", "--max-steps", "0"]) == 1
    assert "FAIL: train ValueError" in capsys.readouterr().out


def test_record_exists_before_fit(fake: dict[str, Any]) -> None:
    train_runner.run("smoke", max_steps=5)
    start = FakeTrainer.instances[0].record_at_start
    assert "start_utc" in start and "end_utc" not in start and start["max_steps"] == 5


def test_interrupt_saves_and_records(fake: dict[str, Any]) -> None:
    FakeTrainer.fail_with = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        train_runner.run("smoke")
    (tr,) = FakeTrainer.instances
    assert tr.saved == [7] and tr.finished  # saved at the interrupt, logs flushed
    (s,) = _record()["sessions"]
    assert s["saved_on_interrupt"] is True and s["steps_done"] == 7


def test_failed_attempt_kept_and_logs_flushed(fake: dict[str, Any]) -> None:
    FakeTrainer.fail_with = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError):
        train_runner.run("smoke")
    assert FakeTrainer.instances[0].finished and FakeTrainer.instances[0].saved == []
    FakeTrainer.fail_with = None
    train_runner.run("smoke", max_steps=3)  # no checkpoint was left, so a rerun is allowed
    first, second = _record()["sessions"]
    assert first["error"].startswith("RuntimeError") and second["steps_done"] == 3
