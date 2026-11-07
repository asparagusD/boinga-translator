"""Tests for train_runner.run() and `boinga train`: TrainerArgs, step stop, boinga_run.json (fakes)."""

from __future__ import annotations

import json
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
        n = len(FakeTrainer.instances)  # coqui dates each folder, so attempts never collide
        self.output_path = Path(str(output_path)) / f"smoke-November-08-2026_10+0{n}AM-abc1234"
        self.output_path.mkdir(parents=True)
        self.total_steps_done = 0
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


def test_existing_checkpoint_refused(fake: dict[str, Any]) -> None:
    old = paths.run_dir("smoke") / "smoke-old"
    old.mkdir(parents=True)
    (old / "checkpoint_50.pth").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="already has checkpoints"):
        train_runner.run("smoke")
    assert FakeTrainer.instances == []


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
    assert cli.main(["train", "--run", "smoke"]) == 1  # the forced save left a checkpoint
    assert "FAIL: train RuntimeError" in capsys.readouterr().out


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
