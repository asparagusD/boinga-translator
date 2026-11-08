"""Tests for ckpt_sampler.py and `boinga sample` / `boinga candidates` (fake folders, log, synth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boinga_translator import ckpt_sampler, cli, paths, synth_basic, train_config

TEXTS = [f"কথা {i}" for i in range(5)]


def _train(step: int) -> str:
    return (f"\n\x1b[1m   --> TIME: 2026-11-08 -- STEP: {step % 400}/400 -- GLOBAL_STEP: {step}"
            f"\x1b[0m\n     | > loss_mel: 30.0  (30.0)\n     | > step_time: 0.3  (0.3)\n")


def _eval(mel: float, l0: float, l1: float) -> str:
    return ("\n\x1b[1m   --> EVAL PERFORMANCE\x1b[0m\n     | > avg_loader_time: 0.01  (+0.0)\n"
            f"     | > avg_loss_0: {l0}  (-0.1)\n     | > avg_loss_1: {l1}  (-0.2)\n"
            f"     | > avg_loss_mel: {mel}  (-0.3)\n")


LOG = (_train(375) + _eval(25.0, 3.0, 60.0) + _train(775) + _eval(22.0, 2.5, 55.0)
       + _train(1175) + _eval(23.0, 2.6, 57.0))


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake run folder with checkpoints 400/800/1200 + best_model_800, a log, a fake synth."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    cfg = paths.REPO / train_config.CONFIG_REL
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"test_sentences": [[t] for t in TEXTS]}), encoding="utf-8")
    folder = paths.run_dir("run1") / "run1-November-08-2026_10+00AM-abc1234"
    folder.mkdir(parents=True)
    for name in ("checkpoint_400.pth", "checkpoint_800.pth", "checkpoint_1200.pth",
                 "best_model_800.pth", "best_model.pth", "config.json"):
        (folder / name).write_bytes(b"x")
    (folder / ckpt_sampler.LOG_FILE).write_text(LOG, encoding="utf-8")
    base = paths.pretrained() / "male"
    base.mkdir(parents=True)
    monkeypatch.setattr(train_config, "base_voice", lambda: "male")
    loaded: list[str] = []

    def load_synth(model: Path, config: Path) -> str:
        loaded.append(Path(model).name)
        return Path(model).name

    def say(synth: str, text: str, out_wav: Path, seed: int = 0) -> float:
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(out_wav).write_bytes(b"RIFF")
        return 1.0 + (TEXTS.index(text) * 0.5 if text in TEXTS else 0.0)

    monkeypatch.setattr(synth_basic, "load_synth", load_synth)
    monkeypatch.setattr(synth_basic, "say", say)
    return {"folder": folder, "loaded": loaded}


def test_parse_eval_takes_the_last_train_step() -> None:
    evals = ckpt_sampler.parse_eval(LOG)
    assert [e["step"] for e in evals] == [375, 775, 1175]
    assert evals[1]["loss_mel"] == 22.0 and evals[1]["loss_total"] == pytest.approx(57.5)


def test_checkpoints_one_per_step(run: dict[str, Any]) -> None:
    ckpts = ckpt_sampler.checkpoints("run1")
    assert {s: p.name for s, p in ckpts.items()} == {
        400: "checkpoint_400.pth", 800: "checkpoint_800.pth", 1200: "checkpoint_1200.pth"}


def test_sample_once_with_pretrained(run: dict[str, Any],
                                     capsys: pytest.CaptureFixture[str]) -> None:
    assert ckpt_sampler.sample("run1") == 3
    out = capsys.readouterr().out
    assert "PASS: sampled 3 checkpoints" in out and "step 800: mean 2.00 s" in out
    root = paths.eval_dir() / "run1"
    assert sorted(p.name for p in root.iterdir()) == [
        "step_0_pretrained", "step_1200", "step_400", "step_800"]
    assert sorted(p.name for p in (root / "step_400").glob("*.wav")) == [
        f"{i}.wav" for i in range(5)]
    done = json.loads((root / "step_800" / ckpt_sampler.DONE).read_text(encoding="utf-8"))
    assert done["file"] == "checkpoint_800.pth" and len(done["durations"]) == 5
    assert run["loaded"][0] == "model.pth"  # pretrained base voice first
    run["loaded"].clear()
    assert ckpt_sampler.sample("run1") == 0  # every step done: nothing reloaded
    assert run["loaded"] == [] and "PASS: sampled 0 checkpoints" in capsys.readouterr().out


def test_sample_needs_a_run(run: dict[str, Any]) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        ckpt_sampler.sample("other")


def test_candidates_sorted_by_loss_mel(run: dict[str, Any],
                                       capsys: pytest.CaptureFixture[str]) -> None:
    rows = ckpt_sampler.candidates("run1")
    assert [r["step"] for r in rows] == [800, 1200, 400]
    assert rows[0]["is_best_model"] is True and rows[1]["is_best_model"] is False
    assert rows[0]["eval_step"] == 775 and rows[2]["loss_mel"] == 25.0
    text = (paths.REPO / "reports" / "candidates_run1.md").read_text(encoding="utf-8")
    assert "| 800 | checkpoint_800.pth | 775 | 22.0 | 57.5 | yes |" in text
    out = capsys.readouterr().out
    assert "=== BEGIN reports/candidates_run1.md ===" in out and "PASS: candidates listed" in out


def test_players_default_and_given(run: dict[str, Any]) -> None:
    ckpt_sampler.sample("run1")
    pairs = ckpt_sampler.players("run1")
    steps = [label.split(" #")[0] for _, label in pairs]
    assert steps[:5] == ["pretrained male"] * 5
    assert list(dict.fromkeys(steps)) == ["pretrained male", "step 800", "step 1200",
                                          "step 400"]  # 4 best by loss_mel + last (1200)
    assert pairs[5][1] == f"step 800 #0: {TEXTS[0]}" and pairs[5][0].endswith("0.wav")
    only = ckpt_sampler.players("run1", steps=[400])
    assert [label.split(" #")[0] for _, label in only] == ["step 400"] * 5


def test_cli(run: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["sample", "--run", "run1"]) == 0
    assert cli.main(["candidates", "--run", "run1"]) == 0
    assert cli.main(["sample", "--run", "nope"]) == 1
    assert "FAIL: sample FileNotFoundError" in capsys.readouterr().out


def test_no_log_and_changed_texts(run: dict[str, Any]) -> None:
    (run["folder"] / ckpt_sampler.LOG_FILE).unlink()
    rows = ckpt_sampler.candidates("run1")
    assert [r["loss_mel"] for r in rows] == [None] * 3  # no eval yet: sorted by step
    ckpt_sampler.sample("run1")
    assert [label.split(" #")[0] for _, label in ckpt_sampler.players("run1")][::5] == [
        "pretrained male", "step 1200"]  # no losses: pretrained + last only
    cfg = paths.REPO / train_config.CONFIG_REL
    cfg.write_text(json.dumps({"test_sentences": [["নতুন"]]}), encoding="utf-8")
    assert ckpt_sampler.sample("run1") == 3  # other texts: every step is sampled again


def test_retrained_run_resamples(run: dict[str, Any]) -> None:
    ckpt_sampler.sample("run1")
    new = run["folder"].with_name("run1-November-09-2026_09+00AM-def5678")
    run["folder"].rename(new)  # same steps, another training of the run
    run["loaded"].clear()
    assert ckpt_sampler.sample("run1") == 3  # old audio is never reused
    assert "model.pth" not in run["loaded"]  # the pretrained voice is still the same
