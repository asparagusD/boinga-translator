"""Tests for synth.VoiceBank: available voices, lazy cached loading, prepared text, seed, errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from boinga_translator import (
    paths,
    pretrained,
    settings,
    synth,
    synth_basic,
    text_frontend,
    train_config,
)

SR = 22050


class FakeSynth:
    """coqui-like Synthesizer: random output (so the seed matters), records the texts it read."""

    def __init__(self, model: Path, config: Path) -> None:
        self.model, self.config = model, config
        self.output_sample_rate = SR
        self.texts: list[str] = []

    def tts(self, text: str) -> list[float]:
        self.texts.append(text)
        return (np.random.rand(len(text) * 100) * 0.1).tolist()

    def save_wav(self, wav: Any, path: str) -> None:
        sf.write(path, np.asarray(wav, dtype=np.float32), self.output_sample_rate)


class FakeLoader:
    """Stands in for synth_basic.load_synth: counts loads, keeps the synths it made."""

    def __init__(self) -> None:
        self.made: list[FakeSynth] = []

    def __call__(self, model: Path, config: Path) -> FakeSynth:
        self.made.append(FakeSynth(model, config))
        return self.made[-1]


def _voice(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / pretrained.MODEL_FILE).write_bytes(b"weights")
    (folder / pretrained.CONFIG_FILE).write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture
def loader() -> FakeLoader:
    return FakeLoader()


@pytest.fixture
def bank(tmp_path: Path, loader: FakeLoader) -> synth.VoiceBank:
    voices = {"baseline": _voice(tmp_path / "male"), "boinga": _voice(tmp_path / "best")}
    return synth.VoiceBank(loader=loader, voices=voices)


def test_default_voices_are_base_dir_and_best(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    _voice(train_config.base_dir())
    made = synth.VoiceBank(loader=FakeLoader())
    assert made.available() == ["baseline"]  # nothing promoted yet
    _voice(paths.best_ckpt())
    assert made.available() == ["baseline", "boinga"]


def test_default_loader_is_synth_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLoader()
    monkeypatch.setattr(synth_basic, "load_synth", fake)
    made = synth.VoiceBank(voices={"baseline": _voice(tmp_path / "male")})
    made.speak("baseline", "আমি")
    assert len(fake.made) == 1


def test_available_needs_model_and_config(tmp_path: Path) -> None:
    half = tmp_path / "best"
    half.mkdir()
    (half / pretrained.MODEL_FILE).write_bytes(b"weights")  # no config.json
    made = synth.VoiceBank(loader=FakeLoader(),
                           voices={"baseline": _voice(tmp_path / "male"), "boinga": half})
    assert made.available() == ["baseline"]


def test_speak_loads_each_voice_once(bank: synth.VoiceBank, loader: FakeLoader,
                                     tmp_path: Path) -> None:
    bank.speak("boinga", "আমি")
    bank.speak("boinga", "তুমি")
    assert len(loader.made) == 1
    assert loader.made[0].model == tmp_path / "best" / pretrained.MODEL_FILE
    assert loader.made[0].config == tmp_path / "best" / pretrained.CONFIG_FILE
    bank.speak("baseline", "আমি")
    assert len(loader.made) == 2


def test_speak_returns_rate_and_float32(bank: synth.VoiceBank) -> None:
    rate, wav = bank.speak("baseline", "আমি ভালা আছি।")
    assert rate == SR
    assert wav.dtype == np.float32 and wav.ndim == 1 and wav.size > 0


def test_speak_reads_prepared_text(bank: synth.VoiceBank, loader: FakeLoader) -> None:
    raw = "  \"আমি\"   ভালা\nআছি।  "  # quotes dropped, whitespace collapsed, danda -> "."
    bank.speak("boinga", raw)
    assert loader.made[0].texts == [text_frontend.prepare_text(raw)]
    assert loader.made[0].texts[0] == "আমি ভালা আছি."


def test_speak_is_seeded(bank: synth.VoiceBank) -> None:
    _, a = bank.speak("baseline", "আমি")
    _, b = bank.speak("baseline", "আমি")
    _, c = bank.speak("baseline", "আমি", seed=settings.SEED + 1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("text", ["", "   \n", "\"()\""])
def test_empty_text_refused_before_loading(bank: synth.VoiceBank, loader: FakeLoader,
                                           text: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        bank.speak("baseline", text)
    assert loader.made == []


def test_unknown_voice_refused(bank: synth.VoiceBank, loader: FakeLoader) -> None:
    with pytest.raises(ValueError, match="unknown voice 'female'"):
        bank.speak("female", "আমি")
    with pytest.raises(ValueError, match="unknown voice"):  # voice checked before the text
        bank.speak("female", "")
    assert loader.made == []


def test_missing_boinga_refused(tmp_path: Path) -> None:
    made = synth.VoiceBank(loader=FakeLoader(),
                           voices={"baseline": _voice(tmp_path / "male"),
                                   "boinga": tmp_path / "best"})
    with pytest.raises(ValueError, match="not available"):
        made.speak("boinga", "আমি")


def test_speak_to_file_writes_wav_and_returns_duration(bank: synth.VoiceBank,
                                                       tmp_path: Path) -> None:
    out = tmp_path / "out" / "a.wav"
    seconds = bank.speak_to_file("boinga", "আমি ভালা আছি।", out)
    data, rate = sf.read(out)
    assert rate == SR
    assert seconds == pytest.approx(len(data) / SR)
    _, wav = bank.speak("boinga", "আমি ভালা আছি।")
    assert len(data) == wav.size  # same seed, same audio length
