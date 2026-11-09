"""Tests for evaluate.generate and `boinga eval-generate` with a fake VoiceBank (no coqui)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import cli, evaluate, paths

RATE = 22050


class FakeBank:
    """VoiceBank stand-in: a tone whose length grows with the text; records each call."""

    def __init__(self, voices: tuple[str, ...] = ("baseline", "boinga"),
                 level: float = 0.3) -> None:
        self.voices, self.level, self.calls = voices, level, []

    def available(self) -> list[str]:
        return list(self.voices)

    def speak(self, voice: str, text: str, seed: int = 0) -> tuple[int, np.ndarray]:
        self.calls.append((voice, text))
        n = int(RATE * (1.0 + 0.05 * len(text) + (0.5 if voice == "boinga" else 0.0)))
        return RATE, (self.level * np.sin(np.arange(n) * 0.05)).astype(np.float32)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 3,
           refs: tuple[bool, ...] | None = None) -> dict[str, Path]:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(evaluate, "_target_rate", lambda: RATE)
    refs = refs or (True,) * n
    prompts, wavs = [], {}
    for i, ref in enumerate(refs, 1):
        cid = str(100 + i)
        prompts.append({"id": f"P{i:02d}", "clip_id": cid, "ctg_text": f" আঁই যাই {i} ",
                        "bn_text": "আমি যাই", "has_reference": ref})
        wav = tmp_path / "corpus" / f"{cid}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(16000 * 2) / 16000  # 2 s at 16 kHz, stereo: the transform must fix both
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        sf.write(wav, np.stack([tone, tone], axis=1), 16000)
        wavs[cid] = wav
    cfg = paths.configs_dir() / "eval_prompts.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    return wavs


def test_generate_writes_sets_index_and_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    wavs = _setup(tmp_path, monkeypatch, refs=(True, True, False))
    bank = FakeBank()
    counts = evaluate.generate(bank=bank, wav_paths=wavs)
    assert counts == {"A": 3, "B": 3, "R": 2}
    final = paths.eval_dir() / "final"
    for name in ("P01", "P02", "P03"):
        assert (final / "A_baseline" / f"{name}.wav").is_file()
        assert (final / "B_boinga" / f"{name}.wav").is_file()
    assert not (final / "R_real" / "P03.wav").exists()  # no reference for P03
    data, sr = sf.read(final / "R_real" / "P01.wav")
    assert sr == RATE and data.ndim == 1  # resampled, mono
    index = pd.read_csv(final / "index.csv", dtype=str)
    assert list(index.columns) == ["id", "set", "path", "duration_s", "rms_db", "text"]
    assert len(index) == 8
    assert sorted(index["set"].unique()) == ["A_baseline", "B_boinga", "R_real"]
    assert set(index["text"]) == {" আঁই যাই 1 ", " আঁই যাই 2 ", " আঁই যাই 3 "}  # verbatim
    assert ("baseline", " আঁই যাই 1 ") in bank.calls and ("boinga", " আঁই যাই 3 ") in bank.calls
    out = capsys.readouterr().out
    assert "=== BEGIN reports/eval_generation.md ===" in out
    report = (paths.REPO / "reports" / "eval_generation.md").read_text(encoding="utf-8")
    assert "| A_baseline | 3 |" in report and "| R_real | 2 |" in report


def test_generate_skips_done_and_redoes_changed_prompts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wavs = _setup(tmp_path, monkeypatch)
    evaluate.generate(bank=FakeBank(), wav_paths=wavs)
    again = FakeBank()
    assert evaluate.generate(bank=again, wav_paths=wavs) == {"A": 3, "B": 3, "R": 3}
    assert again.calls == []  # every wav already matches its prompt
    cfg = paths.configs_dir() / "eval_prompts.json"
    prompts = json.loads(cfg.read_text(encoding="utf-8"))
    prompts[1]["ctg_text"] = "হেতে আইয়ে"  # P02 replaced by --exclude
    cfg.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    third = FakeBank()
    evaluate.generate(bank=third, wav_paths=wavs)
    assert third.calls == [("baseline", "হেতে আইয়ে"), ("boinga", "হেতে আইয়ে")]
    index = pd.read_csv(paths.eval_dir() / "final" / "index.csv", dtype=str)
    assert set(index.loc[index["id"] == "P02", "text"]) == {"হেতে আইয়ে"}


def test_generate_refuses_missing_voice_and_bad_audio(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wavs = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="voice 'boinga' not available"):
        evaluate.generate(bank=FakeBank(voices=("baseline",)), wav_paths=wavs)
    with pytest.raises(ValueError, match="too short, quiet or not finite"):
        evaluate.generate(bank=FakeBank(level=1e-4), wav_paths=wavs)  # ~-83 dBFS
    with pytest.raises(KeyError, match="no corpus wav for clip 101"):
        evaluate.generate(bank=FakeBank(), wav_paths={})


def test_players_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                         capsys: pytest.CaptureFixture[str]) -> None:
    wavs = _setup(tmp_path, monkeypatch, n=4)
    bank = FakeBank()
    monkeypatch.setattr(evaluate, "_bank", lambda: bank)
    monkeypatch.setattr(evaluate, "_corpus_wavs", lambda: wavs)
    assert cli.main(["eval-generate"]) == 0
    assert "PASS: generated A=4 B=4 R=4" in capsys.readouterr().out
    files, labels = evaluate.players(("P01", "P02", "P03"))
    assert len(files) == len(labels) == 9 and all(Path(f).is_file() for f in files)
    assert labels[:3] == ["P01 A_baseline:  আঁই যাই 1 ", "P01 B_boinga:  আঁই যাই 1 ",
                          "P01 R_real:  আঁই যাই 1 "]
    with pytest.raises(KeyError, match="P09 not in"):
        evaluate.players(("P01", "P09"))
    monkeypatch.setattr(evaluate, "_bank", lambda: FakeBank(voices=()))
    (paths.eval_dir() / "final" / "index.csv").unlink()
    assert cli.main(["eval-generate"]) == 1
    assert "FAIL: eval-generate:" in capsys.readouterr().out
