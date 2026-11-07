"""Baseline smoke synthesis: pretrained voices read configs/baseline_prompts.json for listening."""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from boinga_translator import (
    audit_summary,
    colab_runtime,
    corpus,
    paths,
    pretrained,
    report_io,
    settings,
    synth_basic,
)

PROMPTS_FILE = "baseline_prompts.json"
MIN_SECONDS = 0.3
MIN_DBFS = -50.0


def load_prompts(path: Path | None = None, prefix: str = "") -> dict[str, str]:
    """Prompt key -> text, in file order; skips null prompts and "_" metadata such as _sources."""
    raw = json.loads((path or paths.configs_dir() / PROMPTS_FILE).read_text(encoding="utf-8"))
    return {key: text for key, text in raw.items()
            if isinstance(text, str) and not key.startswith("_") and key.startswith(prefix)}


# --- Chittagonian prompts from the corpus (C3.6) ---------------------------------------------

CTG_KEYS = ("CTG_1", "CTG_2", "CTG_3")
CTG_WORDS = (5, 12)  # preferred word count of a dialect prompt


def bangla_similarity(ctg_nfc: str, bn_text: str) -> float:
    """difflib ratio of the dialect line to its Standard Bangla translation (low = dialectal)."""
    return difflib.SequenceMatcher(None, ctg_nfc, corpus.normalise_text(bn_text)).ratio()


def pick_ctg_prompts(audit_df: pd.DataFrame, index_df: pd.DataFrame,
                     seed: int = settings.SEED) -> dict[str, tuple[str, str]]:
    """CTG_1..3 -> (clip_id, ctg_text verbatim): clean test-split clips, 5-12 words preferred,
    least like their Bangla translation first; ties broken by a seeded shuffle."""
    test = audit_df[audit_summary.clean_mask(audit_df)
                    & (audit_df["split_official"] == "test")]
    test = test.merge(index_df[["clip_id", "ctg_text", "bn_text"]], on="clip_id")
    if len(test) < len(CTG_KEYS):
        raise ValueError(f"need {len(CTG_KEYS)} clean test-split clips, found {len(test)}")
    words = test["ctg_text_nfc"].str.split().str.len()
    in_range = test[words.between(*CTG_WORDS)]
    if len(in_range) < len(CTG_KEYS):
        print(f"WARN: only {len(in_range)} clean test clips with {CTG_WORDS[0]}-{CTG_WORDS[1]} "
              "words; picking from all clean test clips")
        in_range = test
    ranked = in_range.sample(frac=1, random_state=seed).assign(
        sim=lambda d: [bangla_similarity(c, b) for c, b in zip(d["ctg_text_nfc"], d["bn_text"],
                                                               strict=True)])
    top = ranked.sort_values("sim", kind="stable").head(len(CTG_KEYS))
    return {key: (row.clip_id, row.ctg_text)
            for key, row in zip(CTG_KEYS, top.itertuples(), strict=True)}


def write_ctg_prompts(picks: Mapping[str, tuple[str, str]], path: Path | None = None) -> Path:
    """Set CTG_* and "_sources" {key: clip_id} in baseline_prompts.json (other keys kept); emit."""
    path = path or paths.configs_dir() / PROMPTS_FILE
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    raw.update({key: text for key, (_, text) in picks.items()})
    raw["_sources"] = {key: clip_id for key, (clip_id, _) in picks.items()}
    rel = f"configs/{PROMPTS_FILE}"
    repo_path, _ = report_io.write_report(json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                                          rel)
    report_io.emit(repo_path, rel)
    return repo_path


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
