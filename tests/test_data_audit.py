"""Tests for data_audit.py: audio_stats on synthetic wavs, text_stats and the model charset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from boinga_translator import data_audit

SR = 16000
KEYS = {"decodable", "error", "sample_rate", "channels", "subtype", "duration_s", "peak",
        "clip_ratio", "rms_db", "lead_silence_s", "trail_silence_s"}


def _sine(seconds: float, amp: float, sr: int = SR, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype("float32")


def _write(path: Path, data: np.ndarray, sr: int = SR) -> Path:
    sf.write(path, data, sr, subtype="PCM_16")
    return path


def test_sine_with_leading_and_trailing_silence(tmp_path: Path) -> None:
    y = np.concatenate([np.zeros(int(0.5 * SR), "float32"), _sine(1.0, 0.5),
                        np.zeros(int(0.25 * SR), "float32")])
    s = data_audit.audio_stats(_write(tmp_path / "a.wav", y))
    assert set(s) == KEYS
    assert s["decodable"] is True and s["error"] is None
    assert (s["sample_rate"], s["channels"], s["subtype"]) == (SR, 1, "PCM_16")
    assert s["duration_s"] == pytest.approx(1.75)
    assert s["peak"] == pytest.approx(0.5, abs=1e-3)
    assert s["clip_ratio"] == 0.0
    # rms of a 0.5 sine is 0.5/sqrt(2); over 1.0 s of 1.75 s
    assert s["rms_db"] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2) * np.sqrt(1 / 1.75)),
                                        abs=0.1)
    frame = 2048 / SR  # trim's centred frames bleed up to half a frame into the silence
    assert s["lead_silence_s"] == pytest.approx(0.5, abs=frame)
    assert s["trail_silence_s"] == pytest.approx(0.25, abs=frame)
    assert s["lead_silence_s"] <= 0.5 and s["trail_silence_s"] <= 0.25  # never overestimated


def test_digital_silence_is_all_silence(tmp_path: Path) -> None:
    s = data_audit.audio_stats(_write(tmp_path / "s.wav", np.zeros(SR, "float32")))
    assert s["peak"] == 0.0 and s["clip_ratio"] == 0.0
    assert s["rms_db"] == data_audit.RMS_FLOOR_DB
    assert s["lead_silence_s"] == s["trail_silence_s"] == pytest.approx(1.0)


def test_clipped_sine(tmp_path: Path) -> None:
    s = data_audit.audio_stats(_write(tmp_path / "c.wav", np.clip(_sine(1.0, 2.0), -1, 1)))
    assert s["peak"] >= 0.999
    assert 0.5 < s["clip_ratio"] < 0.8  # |2 sin| >= 1 for 2/3 of a period
    assert s["lead_silence_s"] == 0.0


def test_stereo_uses_a_mono_mix_for_silence(tmp_path: Path) -> None:
    left = np.concatenate([np.zeros(int(0.5 * SR), "float32"), _sine(0.5, 0.5)])
    data = np.stack([left, np.zeros_like(left)], axis=1)
    s = data_audit.audio_stats(_write(tmp_path / "st.wav", data))
    assert s["channels"] == 2
    assert s["duration_s"] == pytest.approx(1.0)
    assert s["lead_silence_s"] == pytest.approx(0.5, abs=2048 / SR)
    assert s["peak"] == pytest.approx(0.5, abs=1e-3)


def test_antiphase_stereo_is_silence(tmp_path: Path) -> None:
    left = _sine(0.5, 0.5)
    path = tmp_path / "ap.wav"  # FLOAT: PCM_16 rounds x and -x asymmetrically (1-LSB residue)
    sf.write(path, np.stack([left, -left], axis=1), SR, subtype="FLOAT")
    s = data_audit.audio_stats(path)
    assert s["peak"] == pytest.approx(0.5, abs=1e-3)
    assert s["rms_db"] == data_audit.RMS_FLOOR_DB
    assert s["lead_silence_s"] == s["trail_silence_s"] == pytest.approx(0.5)


def test_float_wav_above_full_scale_and_non_finite(tmp_path: Path) -> None:
    loud = tmp_path / "loud.wav"
    sf.write(loud, _sine(0.5, 1.5), SR, subtype="FLOAT")
    s = data_audit.audio_stats(loud)
    assert s["peak"] == pytest.approx(1.5, abs=1e-3) and s["clip_ratio"] > 0.5
    bad = tmp_path / "nan.wav"
    y = _sine(0.5, 0.5)
    y[100] = np.nan
    sf.write(bad, y, SR, subtype="FLOAT")
    s = data_audit.audio_stats(bad)
    assert s["decodable"] is True and s["error"] == "non-finite samples (NaN/inf)"
    assert s["duration_s"] == pytest.approx(0.5) and s["peak"] is None


def test_clip_shorter_than_a_frame(tmp_path: Path) -> None:
    s = data_audit.audio_stats(_write(tmp_path / "t.wav", _sine(0.01, 0.5)))
    assert s["decodable"] is True and s["error"] is None
    assert type(s["lead_silence_s"]) is float and type(s["rms_db"]) is float


def test_8khz_file(tmp_path: Path) -> None:
    s = data_audit.audio_stats(_write(tmp_path / "n.wav", _sine(2.0, 0.3, sr=8000), sr=8000))
    assert s["sample_rate"] == 8000
    assert s["duration_s"] == pytest.approx(2.0)


def test_empty_wav(tmp_path: Path) -> None:
    s = data_audit.audio_stats(_write(tmp_path / "e.wav", np.zeros(0, "float32")))
    assert s["decodable"] is True
    assert s["duration_s"] == 0.0 and s["peak"] == 0.0
    assert s["lead_silence_s"] == s["trail_silence_s"] == 0.0


# --- text_stats / load_model_charset ---------------------------------------------------------

TEXT_KEYS = set(data_audit.TEXT_KEYS)
FAKE = frozenset("আমিবাজারেযাব ।")  # no ঁ, no digits, no Latin


def test_text_stats_reports_char_missing_from_charset() -> None:
    text = "আঁমি বাজারে যাব।"
    s = data_audit.text_stats(text, FAKE)
    assert set(s) == TEXT_KEYS
    assert s["n_chars"] == len(text) and s["n_words"] == 3 and s["empty"] is False
    assert s["chars_not_in_model"] == ["ঁ"]
    assert s["oov_ratio"] == pytest.approx(1 / 14)  # 14 non-whitespace code points
    assert s["has_latin"] is False and s["has_digits"] is False


def test_text_stats_empty_and_whitespace_only() -> None:
    for text in ("", "  \t\n"):
        s = data_audit.text_stats(text, FAKE)
        assert set(s) == TEXT_KEYS and s["n_chars"] == len(text)
        assert s["empty"] is True and s["n_words"] == 0
        assert s["chars_not_in_model"] == [] and s["oov_ratio"] == 0.0


def test_text_stats_mixed_latin_and_digits() -> None:
    text = "আমি OK বাজারে ২টা café 3"
    s = data_audit.text_stats(text, FAKE)
    assert s["has_latin"] is True and s["has_digits"] is True
    assert s["chars_not_in_model"] == sorted(set(text.replace(" ", "")) - FAKE)
    assert {"O", "K", "২", "ট", "é", "3"} <= set(s["chars_not_in_model"])
    assert " " not in s["chars_not_in_model"]
    assert data_audit.text_stats("১০", FAKE)["has_digits"] is True  # Bengali digits
    assert data_audit.text_stats("é", FAKE)["has_latin"] is True
    assert data_audit.text_stats("Ａ", FAKE)["has_latin"] is True  # fullwidth A


def test_text_stats_never_modifies_or_normalises() -> None:
    decomposed = "য়"  # য + nukta; the charset holds only the precomposed য়
    s = data_audit.text_stats(decomposed, frozenset("য়"))
    assert s["chars_not_in_model"] == ["য", "়"]  # measured as given, not NFC'd
    assert s["n_chars"] == 2


def test_load_model_charset_from_facts(tmp_path: Path) -> None:
    facts = tmp_path / "facts.json"
    facts.write_text('{"female": {"characters": "অআ‌", "punctuations": ".,", '
                     '"eos": "।"}}', encoding="utf-8")
    assert data_audit.load_model_charset("female", facts) == frozenset("অআ‌., ")
    with pytest.raises(ValueError, match=r"no voice 'male'"):
        data_audit.load_model_charset("male", facts)


def test_load_model_charset_real_facts() -> None:
    cs = data_audit.load_model_charset("female")
    assert {"অ", "য়", "়", " ", ","} <= cs
    assert "।" not in cs  # the eos symbol is not in characters/punctuations


@pytest.mark.parametrize("make", [
    lambda p: p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEgarbage"),
    lambda p: None,  # missing file
])
def test_unreadable_never_raises(tmp_path: Path, make: Callable[[Path], object]) -> None:
    path = tmp_path / "bad.wav"
    make(path)
    s = data_audit.audio_stats(path)
    assert set(s) == KEYS
    assert s["decodable"] is False
    assert isinstance(s["error"], str) and s["error"]
    assert s["duration_s"] is None and s["peak"] is None
