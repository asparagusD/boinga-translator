"""Baseline smoke synthesis: pretrained voices read configs/baseline_prompts.json for listening."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from boinga_translator import colab_runtime, paths, pretrained, synth_basic

PROMPTS_FILE = "baseline_prompts.json"
MIN_SECONDS = 0.3
MIN_DBFS = -50.0


def load_prompts(path: Path | None = None) -> dict[str, str]:
    """Prompt key -> text, in file order; null prompts (not written yet) are skipped."""
    raw = json.loads((path or paths.configs_dir() / PROMPTS_FILE).read_text(encoding="utf-8"))
    return {key: text for key, text in raw.items() if text is not None}


def out_dir(voice: str) -> Path:
    """Drive folder of one voice's baseline wavs."""
    return paths.eval_dir() / "baseline_smoke" / voice


def rms_dbfs(wav: np.ndarray) -> float:
    """RMS level in dB relative to full scale (1.0); silence gives about -200."""
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-10)))


def _model_charset(voice: str) -> set[str] | None:
    """Characters + punctuation + space of the cached voice's config.

    None (check skipped) if the voice is not cached or its config lists no characters. The check
    is on raw text, before coqui's cleaner, so it is a hint, not the model's exact input.
    """
    config = paths.pretrained() / voice / pretrained.CONFIG_FILE
    if not config.is_file():
        return None
    chars = json.loads(config.read_text(encoding="utf-8")).get("characters") or {}
    if not chars.get("characters"):
        return None
    return set(chars["characters"] + (chars.get("punctuations") or "") + " ")


def run(voice: str, synth: Any = None, prompts: Mapping[str, str] | None = None,
        charset: set[str] | None = None) -> bool:
    """Synthesise every prompt for one voice; PASS if each wav is > 0.3 s and > -50 dBFS.

    Re-running overwrites the wavs (same seed), so the cell is safe to repeat.
    """
    if synth is None:
        files = pretrained.fetch(voice)
        synth = synth_basic.load_synth(files["model"], files["config"])
    prompts = load_prompts() if prompts is None else prompts
    charset = _model_charset(voice) if charset is None else charset
    bad = []
    for key, text in prompts.items():
        if charset is not None and (unknown := sorted(set(text) - charset)):
            listed = ", ".join(f"{c} (U+{ord(c):04X})" for c in unknown)
            print(f"WARN: {voice} {key}: not in model charset: {listed}")
        wav_path = out_dir(voice) / f"{key}.wav"
        seconds = synth_basic.say(synth, text, wav_path)
        level = rms_dbfs(sf.read(wav_path, dtype="float32")[0])
        print(f"{voice} {key} {seconds:.2f} s {level:.1f} dBFS")
        if seconds <= MIN_SECONDS or level <= MIN_DBFS:
            bad.append(key)
    if bad:
        print(f"FAIL: baseline {voice} {' '.join(bad)}")
        return False
    print(f"PASS: baseline {voice} ({len(prompts)} wavs in {out_dir(voice)})")
    return True


def players(voice: str, prompts: Mapping[str, str] | None = None) -> list[tuple[Path, str]]:
    """(wav, "<voice> <key>: <text>") for each prompt whose wav exists, for colab_runtime.play."""
    prompts = load_prompts() if prompts is None else prompts
    pairs = [(out_dir(voice) / f"{key}.wav", f"{voice} {key}: {text}")
             for key, text in prompts.items()]
    return [(wav, label) for wav, label in pairs if wav.is_file()]


def play_all(voices: Iterable[str] | None = None, prompts: Mapping[str, str] | None = None) -> None:
    """Show audio players for every voice's baseline wavs, labelled with the prompt text."""
    for voice in pretrained.voices() if voices is None else voices:
        pairs = players(voice, prompts)
        if not pairs:
            print(f"FAIL: no baseline wavs for {voice} in {out_dir(voice)}")
            continue
        colab_runtime.play([w for w, _ in pairs], [label for _, label in pairs])
