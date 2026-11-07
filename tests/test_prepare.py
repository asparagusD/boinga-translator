"""Tests for prepare.py: target_sample_rate from the configs, transform_audio on synthetic wavs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from boinga_translator import prepare

TARGET = 22050
PEAK = 10 ** (prepare.PEAK_DBFS / 20)
LSB = 1 / 32768  # one PCM_16 step
KEYS = {"in_sr", "out_sr", "in_dur", "out_dur", "trimmed_s", "gain_db", "upsampled"}


def _sine(seconds: float, amp: float, sr: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype("float32")


def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * seconds), "float32")


def _write(path: Path, data: np.ndarray, sr: int) -> Path:
    sf.write(path, data, sr, subtype="PCM_16")
    return path


def _lead_zeros(y: np.ndarray) -> int:
    return int(np.argmax(np.abs(y) > 1e-3))


# --- target_sample_rate ----------------------------------------------------------------------


def _configs(tmp_path: Path, voice: str, rate: object) -> tuple[Path, Path]:
    sel = tmp_path / "speaker_selection.json"
    sel.write_text(json.dumps({"base_voice": voice}), encoding="utf-8")
    facts = tmp_path / "pretrained_facts.json"
    facts.write_text(json.dumps({"male": {"sample_rate": rate},
                                 "female": {"sample_rate": 16000}}), encoding="utf-8")
    return sel, facts


def test_target_sample_rate_follows_the_base_voice(tmp_path: Path) -> None:
    sel, facts = _configs(tmp_path, "male", 24000)
    assert prepare.target_sample_rate(sel, facts) == 24000
    sel, facts = _configs(tmp_path, "female", 24000)
    assert prepare.target_sample_rate(sel, facts) == 16000


def test_target_sample_rate_of_the_repo_configs() -> None:
    assert prepare.target_sample_rate() == TARGET  # male base voice, pretrained config


@pytest.mark.parametrize(("voice", "rate"), [("robot", 22050), ("male", None), ("male", 0),
                                             ("male", "22050")])
def test_target_sample_rate_refuses_bad_configs(tmp_path: Path, voice: str, rate: object) -> None:
    sel, facts = _configs(tmp_path, voice, rate)
    with pytest.raises(ValueError):
        prepare.target_sample_rate(sel, facts)


# --- transform_audio -------------------------------------------------------------------------


def test_stereo_is_mixed_to_mono(tmp_path: Path) -> None:
    left = _sine(1.0, 0.5, TARGET)
    src = _write(tmp_path / "st.wav", np.stack([left, 0.5 * left], axis=1), TARGET)
    out = tmp_path / "out" / "st.wav"
    s = prepare.transform_audio(src, out, TARGET)
    assert set(s) == KEYS
    info = sf.info(str(out))
    assert (info.channels, info.samplerate, info.subtype) == (1, TARGET, "PCM_16")
    y, _ = sf.read(str(out), dtype="float32")
    # the mix is 0.75 * left, then peak-normalised: same shape as left
    assert np.corrcoef(y[: left.size], left[: y.size])[0, 1] > 0.999


def test_48k_is_resampled_to_the_target(tmp_path: Path) -> None:
    src = _write(tmp_path / "hi.wav", _sine(1.0, 0.5, 48000), 48000)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert (s["in_sr"], s["out_sr"], s["upsampled"]) == (48000, TARGET, False)
    assert sf.info(str(tmp_path / "o.wav")).samplerate == TARGET
    assert s["in_dur"] == pytest.approx(1.0)
    assert s["out_dur"] == pytest.approx(1.0, abs=0.01)


def test_8k_is_flagged_upsampled(tmp_path: Path) -> None:
    src = _write(tmp_path / "lo.wav", _sine(1.0, 0.5, 8000), 8000)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert (s["in_sr"], s["out_sr"], s["upsampled"]) == (8000, TARGET, True)
    assert sf.info(str(tmp_path / "o.wav")).samplerate == TARGET


def test_at_target_rate_is_not_upsampled(tmp_path: Path) -> None:
    src = _write(tmp_path / "eq.wav", _sine(1.0, 0.5, TARGET), TARGET)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert s["upsampled"] is False and s["in_sr"] == s["out_sr"] == TARGET


def test_edge_silence_is_trimmed_to_a_100ms_pad(tmp_path: Path) -> None:
    y = np.concatenate([_silence(0.5, TARGET), _sine(1.0, 0.5, TARGET), _silence(0.5, TARGET)])
    src = _write(tmp_path / "pad.wav", y, TARGET)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    out, _ = sf.read(str(tmp_path / "o.wav"), dtype="float32")
    pad = int(prepare.PAD_S * TARGET)
    extra = 1024 + 512  # trim's centred frames keep up to ~half a frame (+ one hop) more
    assert pad <= _lead_zeros(out) <= pad + extra  # never less than the 100 ms pad
    assert pad <= _lead_zeros(out[::-1]) <= pad + extra
    assert s["in_dur"] == pytest.approx(2.0)
    assert 1.2 <= s["out_dur"] <= 1.2 + 2 * extra / TARGET
    assert s["trimmed_s"] == pytest.approx(s["in_dur"] - s["out_dur"], abs=1e-6)


@pytest.mark.parametrize("lead", [True, False])
def test_pad_is_clamped_at_the_clip_edges(tmp_path: Path, lead: bool) -> None:
    parts = [_silence(0.03, TARGET), _sine(1.0, 0.5, TARGET)]
    y = np.concatenate(parts if lead else parts[::-1])
    src = _write(tmp_path / "edge.wav", y, TARGET)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert s["trimmed_s"] == 0.0  # less than 100 ms of silence: nothing cut, nothing added
    assert s["out_dur"] == pytest.approx(s["in_dur"])


@pytest.mark.parametrize("amp", [0.05, 0.5, 1.0])
def test_peak_is_normalised_to_minus_1_dbfs(tmp_path: Path, amp: float) -> None:
    src = _write(tmp_path / "a.wav", _sine(1.0, amp, 16000), 16000)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    out, _ = sf.read(str(tmp_path / "o.wav"), dtype="float32")
    peak = float(np.abs(out).max())
    assert peak <= PEAK + LSB  # nothing above -1 dBFS
    assert peak == pytest.approx(PEAK, abs=0.01)
    assert (s["gain_db"] > 0) == (amp < PEAK)


def test_silent_clip_is_refused_and_not_written(tmp_path: Path) -> None:
    src = _write(tmp_path / "z.wav", _silence(1.0, 16000), 16000)
    with pytest.raises(ValueError, match="silent"):
        prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert not (tmp_path / "o.wav").exists()


def test_non_finite_clip_is_refused_and_not_written(tmp_path: Path) -> None:
    y = _sine(1.0, 0.5, 16000)
    y[100] = np.nan
    src = tmp_path / "nan.wav"
    sf.write(src, y, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert not (tmp_path / "o.wav").exists()


def test_empty_clip_is_refused(tmp_path: Path) -> None:
    src = _write(tmp_path / "e.wav", np.zeros(0, "float32"), 16000)
    with pytest.raises(ValueError, match="empty"):
        prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
