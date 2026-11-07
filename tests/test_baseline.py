"""Tests for baseline.py and `boinga baseline`: prompts file, PASS/FAIL rules, charset warning, players."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from boinga_translator import baseline, cli, colab_runtime, paths

SR = 16000


class FakeSynth:
    """Text 'quiet' -> near-silence, 'short' -> 0.1 s, anything else -> 1 s at about -20 dBFS."""

    output_sample_rate = SR

    def tts(self, text: str) -> list[float]:
        n = SR // 10 if text == "short" else SR
        amp = 1e-5 if text == "quiet" else 0.1
        return (amp * np.sin(np.arange(n) / 5.0)).tolist()

    def save_wav(self, wav: list[float], path: str) -> None:
        sf.write(path, np.asarray(wav, dtype=np.float32), SR)


@pytest.fixture(autouse=True)
def _tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))


def test_repo_prompts_file_shape() -> None:
    raw = json.loads((paths.configs_dir() / "baseline_prompts.json").read_text(encoding="utf-8"))
    assert list(raw) == ["STD_1", "STD_2", "STD_3", "CTG_1", "CTG_2", "CTG_3"]
    assert all(raw[k] is None for k in ("CTG_1", "CTG_2", "CTG_3"))
    std = [raw[k] for k in ("STD_1", "STD_2", "STD_3")]
    assert all(unicodedata.is_normalized("NFC", s) for s in std)
    assert all(6 <= len(s.split()) <= 12 for s in std)
    assert sum(s.endswith("?") for s in std) == 1


def test_load_prompts_skips_null(tmp_path: Path) -> None:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps({"A": "ক", "B": None, "C": "খ"}), encoding="utf-8")
    assert baseline.load_prompts(p) == {"A": "ক", "C": "খ"}


def test_rms_dbfs() -> None:
    assert baseline.rms_dbfs(np.full(100, 0.1)) == pytest.approx(-20.0)
    assert baseline.rms_dbfs(np.zeros(10)) < -100


def test_run_pass_writes_wavs(capsys: pytest.CaptureFixture[str]) -> None:
    ok = baseline.run("female", synth=FakeSynth(), prompts={"STD_1": "hello", "STD_2": "world"})
    assert ok
    out_dir = paths.eval_dir() / "baseline_smoke" / "female"
    assert sorted(p.name for p in out_dir.glob("*.wav")) == ["STD_1.wav", "STD_2.wav"]
    out = capsys.readouterr().out
    assert "STD_1 1.00 s" in out and "dBFS" in out
    assert "PASS: baseline female" in out


def test_run_fail_lists_bad_keys(capsys: pytest.CaptureFixture[str]) -> None:
    ok = baseline.run("male", synth=FakeSynth(),
                      prompts={"A": "fine", "B": "quiet", "C": "short"})
    assert not ok
    assert "FAIL: baseline male B C" in capsys.readouterr().out


def test_run_warns_on_out_of_charset_chars(capsys: pytest.CaptureFixture[str]) -> None:
    baseline.run("female", synth=FakeSynth(), prompts={"A": "কx"}, charset=set("ক"))
    assert "WARN: female A: not in model charset: x (U+0078)" in capsys.readouterr().out


def _cache_config(voice: str, characters: object) -> None:
    d = paths.pretrained() / voice
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"characters": characters}), encoding="utf-8")


def test_model_charset_from_cached_config(capsys: pytest.CaptureFixture[str]) -> None:
    _cache_config("female", {"characters": "কখ", "punctuations": "।"})
    assert baseline._model_charset("female") == {"ক", "খ", "।", " "}
    baseline.run("female", synth=FakeSynth(), prompts={"A": "ক গ।"})  # charset read from config
    out = capsys.readouterr().out
    assert "WARN: female A: not in model charset: গ (U+0997)" in out


@pytest.mark.parametrize("characters", [None, {}, {"characters": None, "punctuations": "!"}])
def test_model_charset_skipped_without_characters(characters: object) -> None:
    _cache_config("male", characters)
    assert baseline._model_charset("male") is None


def test_model_charset_not_cached() -> None:
    assert baseline._model_charset("female") is None


def test_play_all_fails_visibly_without_wavs(capsys: pytest.CaptureFixture[str]) -> None:
    baseline.play_all(["female"], prompts={"STD_1": "hi"})
    assert "FAIL: no baseline wavs for female" in capsys.readouterr().out


def test_players_pairs_existing_wavs_with_text() -> None:
    prompts = {"STD_1": "hello", "STD_2": "world"}
    baseline.run("female", synth=FakeSynth(), prompts={"STD_1": "hello"})
    pairs = baseline.players("female", prompts=prompts)
    assert pairs == [(paths.eval_dir() / "baseline_smoke" / "female" / "STD_1.wav",
                      "female STD_1: hello")]  # STD_2 has no wav yet


def test_play_all_calls_colab_play(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline.run("male", synth=FakeSynth(), prompts={"STD_1": "hi"})
    seen: list[tuple[list[Path], list[str]]] = []
    monkeypatch.setattr(colab_runtime, "play", lambda w, labels: seen.append((list(w), list(labels))))
    baseline.play_all(["male"], prompts={"STD_1": "hi"})
    assert seen == [([paths.eval_dir() / "baseline_smoke" / "male" / "STD_1.wav"],
                     ["male STD_1: hi"])]


def test_cli_baseline_both_exit_codes(monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[str] = []
    monkeypatch.setattr(baseline, "run", lambda voice: seen.append(voice) or True)
    assert cli.main(["baseline", "--voice", "both"]) == 0
    assert seen == ["female", "male"]
    monkeypatch.setattr(baseline, "run", lambda voice: voice == "female")
    assert cli.main(["baseline", "--voice", "both"]) == 1

    def boom(voice: str) -> bool:
        raise RuntimeError("no model")
    monkeypatch.setattr(baseline, "run", boom)
    assert cli.main(["baseline", "--voice", "male"]) == 1
    assert "FAIL: baseline male: RuntimeError: no model" in capsys.readouterr().out
