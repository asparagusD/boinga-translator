"""Tests for synth_basic.py: seeding, loader with a fake coqui Synthesizer, say() output and duration."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from boinga_translator import settings, synth_basic

SR = 16000


class FakeModel:
    """Has two of the three inference attributes, like a model missing one."""

    inference_noise_scale = 0.0
    length_scale = 0.0


class FakeSynth:
    """Random output (so seeding matters), coqui-like tts/save_wav/output_sample_rate."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.output_sample_rate = SR
        self.tts_model = FakeModel()

    def tts(self, text: str) -> list[float]:
        return (np.random.rand(len(text) * 800) * 0.1).tolist()

    def save_wav(self, wav: list[float], path: str) -> None:
        sf.write(path, np.asarray(wav, dtype=np.float32), self.output_sample_rate)


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}
    torch = types.ModuleType("torch")
    torch.manual_seed = lambda s: calls.__setitem__("manual_seed", s)  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True, manual_seed_all=lambda s: calls.__setitem__("cuda", s))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.fixture
def fake_coqui(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("TTS.utils.synthesizer")
    mod.Synthesizer = FakeSynth  # type: ignore[attr-defined]
    for name, m in (("TTS", types.ModuleType("TTS")), ("TTS.utils", types.ModuleType("TTS.utils")),
                    ("TTS.utils.synthesizer", mod)):
        monkeypatch.setitem(sys.modules, name, m)


def test_set_seed_seeds_torch_and_cuda(fake_torch: dict[str, object]) -> None:
    synth_basic.set_seed(42)
    assert fake_torch == {"manual_seed": 42, "cuda": 42}
    a = np.random.rand(3)
    synth_basic.set_seed(42)
    assert np.array_equal(a, np.random.rand(3))


def test_set_seed_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError
    synth_basic.set_seed()  # must not raise


def test_load_synth_sets_scales_and_logs(
        fake_coqui: None, fake_torch: dict[str, object],
        capsys: pytest.CaptureFixture[str]) -> None:
    synth = synth_basic.load_synth(Path("m.pth"), Path("c.json"))
    assert synth.kwargs == {"tts_checkpoint": "m.pth", "tts_config_path": "c.json",
                            "use_cuda": True}  # default from torch.cuda.is_available()
    assert synth.tts_model.inference_noise_scale == settings.INFER_NOISE_SCALE
    assert synth.tts_model.length_scale == settings.INFER_LENGTH_SCALE
    assert not hasattr(synth.tts_model, "inference_noise_scale_dp")
    out = capsys.readouterr().out
    assert f"inference_noise_scale={settings.INFER_NOISE_SCALE}" in out
    assert "not on model: inference_noise_scale_dp" in out


def test_load_synth_explicit_cpu(fake_coqui: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # use_cuda given: torch not needed
    assert synth_basic.load_synth("m.pth", "c.json", use_cuda=False).kwargs["use_cuda"] is False


def test_say_same_seed_same_audio_and_duration(tmp_path: Path) -> None:
    synth = FakeSynth()
    a = tmp_path / "deep" / "dir" / "a.wav"  # parents do not exist yet
    b = tmp_path / "b.wav"
    dur = synth_basic.say(synth, "আমি", a, seed=7)
    synth_basic.say(synth, "আমি", b, seed=7)
    wa, sra = sf.read(a)
    wb, _ = sf.read(b)
    assert sra == SR and np.array_equal(wa, wb)
    assert dur == pytest.approx(3 * 800 / SR)
    assert dur == pytest.approx(len(wa) / sra)


def test_say_different_seed_differs(tmp_path: Path) -> None:
    synth = FakeSynth()
    synth_basic.say(synth, "ক", tmp_path / "a.wav", seed=1)
    synth_basic.say(synth, "ক", tmp_path / "b.wav", seed=2)
    assert not np.array_equal(sf.read(tmp_path / "a.wav")[0], sf.read(tmp_path / "b.wav")[0])
