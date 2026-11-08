"""Tests for tools/reload_check.py and synth.reload_players: import, rows, dBFS, PASS/FAIL (fake bank)."""

from __future__ import annotations

import importlib
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from boinga_translator import paths, prepare, synth

SR = 22050


def _tool() -> types.ModuleType:
    import reload_check

    return reload_check


class FakeBank:
    """VoiceBank stand-in: fixed-length tone at a chosen level per text."""

    def __init__(self, seconds: float = 1.0, amp: float = 0.1,
                 voices: tuple[str, ...] = ("boinga",), fail: bool = False) -> None:
        self.seconds, self.amp, self.voices, self.fail = seconds, amp, list(voices), fail
        self.spoken: list[tuple[str, str]] = []

    def available(self) -> list[str]:
        return self.voices

    def speak(self, voice: str, text: str) -> tuple[int, np.ndarray]:
        self.spoken.append((voice, text))
        if self.fail:
            raise RuntimeError("model load failed")
        t = np.arange(int(self.seconds * SR)) / SR
        return SR, (self.amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Test manifest with 4 ljspeech rows on a fake fast disk."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "vm"))
    path = paths.data_prepared() / prepare.manifest_names()["test"]
    path.parent.mkdir(parents=True)
    path.write_text("".join(f"ctg_{i}|লাইন {i}.|লাইন {i}.\n" for i in range(4)), encoding="utf-8")
    return path


def test_imports_cleanly() -> None:
    mod = importlib.reload(_tool())
    assert callable(mod.main)
    assert mod.MIN_SECONDS == 0.5 and mod.MIN_DBFS == -50.0 and mod.N_ROWS == 3


def test_test_rows_are_the_first_three(manifest: Path) -> None:
    assert _tool().test_rows() == [("ctg_0", "লাইন 0."), ("ctg_1", "লাইন 1."),
                                   ("ctg_2", "লাইন 2.")]


def test_test_rows_refuse_a_short_manifest(manifest: Path) -> None:
    manifest.write_text("ctg_0|a|a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs 3"):
        _tool().test_rows()


def test_rms_dbfs() -> None:
    tool = _tool()
    assert tool.rms_dbfs(np.full(100, 0.1, dtype=np.float32)) == pytest.approx(-20.0)
    assert tool.rms_dbfs(np.zeros(100, dtype=np.float32)) == -math.inf
    assert tool.rms_dbfs(np.zeros(0, dtype=np.float32)) == -math.inf


def test_pass_writes_wavs_and_list(manifest: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bank = FakeBank()
    assert _tool().main(bank) == 0
    assert [v for v, _ in bank.spoken] == ["boinga"] * 3
    out = paths.eval_dir() / "reload_check"
    assert synth.reload_dir() == out
    rows = json.loads((out / synth.RELOAD_LIST).read_text(encoding="utf-8"))
    assert [r["clip_id"] for r in rows] == ["ctg_0", "ctg_1", "ctg_2"]
    for r in rows:
        data, rate = sf.read(r["wav"])
        assert rate == SR and len(data) == SR and r["seconds"] == pytest.approx(1.0)
    assert capsys.readouterr().out.rstrip().endswith("PASS: reload check (3 wavs)")


@pytest.mark.parametrize(("seconds", "amp"), [(0.4, 0.1), (1.0, 0.001)])
def test_short_or_quiet_fails(manifest: Path, capsys: pytest.CaptureFixture[str],
                              seconds: float, amp: float) -> None:
    assert _tool().main(FakeBank(seconds=seconds, amp=amp)) == 1
    out = capsys.readouterr().out
    assert "FAIL: reload check" in out and "PASS" not in out


def test_fails_without_boinga(manifest: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _tool().main(FakeBank(voices=("baseline",))) == 1
    assert "FAIL: reload check: no promoted model" in capsys.readouterr().out


def test_nan_fails(manifest: Path, capsys: pytest.CaptureFixture[str],
                   monkeypatch: pytest.MonkeyPatch) -> None:
    bank = FakeBank()
    good = bank.speak
    monkeypatch.setattr(bank, "speak", lambda v, t: (SR, np.where(
        np.arange(SR) == 5, np.nan, good(v, t)[1]).astype(np.float32)))
    assert _tool().main(bank) == 1
    assert "finite check" in capsys.readouterr().out


def test_speak_error_is_a_fail_line(manifest: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _tool().main(FakeBank(fail=True)) == 1
    assert "FAIL: reload check: RuntimeError: model load failed" in capsys.readouterr().out


def test_old_list_removed_on_rerun(manifest: Path) -> None:
    assert _tool().main(FakeBank()) == 0
    assert _tool().main(FakeBank(fail=True)) == 1
    assert not (synth.reload_dir() / synth.RELOAD_LIST).exists()  # nothing stale to play


def test_reload_players(manifest: Path) -> None:
    assert _tool().main(FakeBank()) == 0
    wavs, labels = synth.reload_players()
    assert wavs == [str(synth.reload_dir() / f"{i}_ctg_{i}.wav") for i in range(3)]
    assert labels == [f"#{i} ctg_{i}: লাইন {i}." for i in range(3)]
