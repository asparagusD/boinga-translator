"""Tests for train_report.py and `boinga throughput`: log parsing, throughput.json, listening."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from boinga_translator import cli, paths, train_config, train_report, train_runner

BOLD, END = "\x1b[1m", "\x1b[0m"  # coqui's console logger also writes these to the log file


def _block(step: int, step_time: float, avg_step: float, loss: float | str = 2.5) -> str:
    return (f"\n{BOLD}   --> TIME: 2026-11-08 10:00:{step % 60:02d} -- STEP: {step}/400 -- "
            f"GLOBAL_STEP: {step}{END}\n"
            f"     | > loss_disc: 2.61  (2.6)\n"
            f"     | > loss_disc_real_0: 0.3  (0.31)\n"
            f"     | > loss: 2.61  (2.6)\n"
            f"     | > loss_gen: 2.4  (2.41)\n"
            f"     | > loss-1: {loss}  ({loss})\n"
            f"     | > grad_norm_0: tensor(12.3, device='cuda:0')  (tensor(11.9))\n"
            f"     | > step_time: {step_time}  ({avg_step})\n"
            f"     | > loader_time: 0.02  (0.03)\n")


LOG = ("INFO - > Restoring from model.pth ...\n" + _block(0, 9.5, 9.5) + _block(25, 0.8, 1.15)
       + "\n   --> EVAL PERFORMANCE\n     | > avg_loader_time: 0.1\n" + _block(50, 0.9, 1.0))


def test_parse_log() -> None:
    entries = train_report.parse_log(LOG)
    assert [e["global_step"] for e in entries] == [0, 25, 50]
    assert entries[1]["values"]["step_time"] == 0.8 and entries[1]["avgs"]["step_time"] == 1.15
    assert entries[2]["values"]["loss_gen"] == 2.4
    assert "grad_norm_0" not in entries[0]["values"]  # tensor(...) text is skipped, not a crash
    assert "avg_loader_time" not in entries[2]["values"]  # eval block is not a train step


def test_parse_log_nan() -> None:
    (entry,) = train_report.parse_log(_block(25, 0.8, 0.8, loss="nan"))
    assert math.isnan(entry["values"]["loss-1"])  # coqui's second-optimizer total loss


def test_summarise_skips_warmup_step() -> None:
    summary = train_report.summarise(train_report.parse_log(LOG))
    assert summary["sec_per_step_mean"] == pytest.approx(0.85)  # steps 25 and 50, not step 0
    assert summary["n_step_samples"] == 2
    assert summary["steps_per_hour"] == pytest.approx(3600 / 0.85, rel=1e-3)
    assert summary["sec_per_step_coqui_avg"] == 1.0  # coqui's running average, last entry
    assert summary["any_nan_loss"] is False
    assert summary["last_losses"] == {"loss_disc": 2.61, "loss_disc_real_0": 0.3,
                                      "loss": 2.61, "loss_gen": 2.4, "loss-1": 2.5}
    assert summary["warmup_only"] is False


def test_summarise_warmup_only() -> None:
    summary = train_report.summarise(train_report.parse_log(_block(0, 9.5, 9.5)))
    assert summary["warmup_only"] is True and summary["sec_per_step_mean"] == 9.5


def test_summarise_nan_and_no_steps() -> None:
    nan = train_report.summarise(train_report.parse_log(_block(25, 0.8, 0.8, loss="nan")))
    assert nan["any_nan_loss"] is True
    with pytest.raises(ValueError, match="no train step"):
        train_report.summarise([])


@pytest.fixture
def run_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake smoke run on Drive: coqui folder (log, config, checkpoint) + boinga_run.json."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    folder = paths.run_dir("smoke") / "smoke-November-08-2026_10+00AM-abc1234"
    folder.mkdir(parents=True)
    (folder / "trainer_0_log.txt").write_text(LOG, encoding="utf-8")
    (folder / "checkpoint_60.pth").write_bytes(b"x")
    (folder / "config.json").write_text(json.dumps({
        "batch_size": 16, "mixed_precision": True, "test_sentences": [["কথা এক"], ["দুই"]]}),
        encoding="utf-8")
    sessions = [{"peak_gpu_memory_gb": 9.1, "env": {"gpu_name": "Tesla T4"}},
                {"peak_gpu_memory_gb": 9.4, "env": {"gpu_name": "Tesla T4"}}]
    (paths.run_dir("smoke") / train_runner.RUN_RECORD).write_text(
        json.dumps({"run": "smoke", "sessions": sessions}), encoding="utf-8")
    return folder


def test_throughput_report(run_folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_report.throughput("smoke") is True
    out = capsys.readouterr().out
    assert "PASS: throughput recorded" in out
    assert f"=== BEGIN {train_report.THROUGHPUT_REL} ===" in out
    rep = json.loads((paths.REPO / train_report.THROUGHPUT_REL).read_text(encoding="utf-8"))
    assert rep["gpu"] == "Tesla T4" and rep["batch_size"] == 16
    assert rep["mixed_precision"] is True and rep["peak_mem_gb"] == 9.4
    assert rep["sec_per_step_mean"] == pytest.approx(0.85) and rep["steps_done"] == 60
    assert rep["any_nan_loss"] is False and rep["last_losses"]["loss_gen"] == 2.4


def test_throughput_nan_fails(run_folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (run_folder / "trainer_0_log.txt").write_text(_block(25, 0.8, 0.8, loss="nan"), "utf-8")
    assert train_report.throughput("smoke") is False
    assert "FAIL: NaN loss" in capsys.readouterr().out
    assert (paths.REPO / train_report.THROUGHPUT_REL).is_file()  # written for diagnosis
    assert cli.main(["throughput", "--run", "smoke"]) == 1


def test_throughput_without_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    assert cli.main(["throughput", "--run", "smoke"]) == 1
    assert "FAIL: throughput" in capsys.readouterr().out


def test_cli_throughput(run_folder: Path) -> None:
    assert cli.main(["throughput", "--run", "smoke"]) == 0


def test_listen_smoke(run_folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voice = paths.pretrained() / "male"
    voice.mkdir(parents=True)
    monkeypatch.setattr(train_config, "base_voice", lambda: "male")
    calls: list[tuple[str, ...]] = []

    def load_synth(model: Path, config: Path) -> tuple[str, str]:
        calls.append(("load", Path(model).name, Path(config).parent.name))
        return (str(model), str(config))

    def say(synth: Any, text: str, out_wav: Path) -> float:
        calls.append(("say", text, Path(out_wav).name))
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(out_wav).write_bytes(b"RIFF")
        return 1.0

    monkeypatch.setattr(train_report.synth_basic, "load_synth", load_synth)
    monkeypatch.setattr(train_report.synth_basic, "say", say)
    wavs, labels = train_report.listen_smoke("smoke")
    assert calls == [("load", "model.pth", "male"), ("say", "কথা এক", "pretrained.wav"),
                     ("load", "checkpoint_60.pth", run_folder.name),
                     ("say", "কথা এক", "step60.wav")]
    assert [Path(w).parent for w in wavs] == [paths.eval_dir() / "smoke"] * 2
    assert labels == ["pretrained male: কথা এক", "smoke step 60: কথা এক"]
