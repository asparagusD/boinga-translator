"""Corpus audit: per-clip audio stats and per-transcript text/charset stats for exclusions."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from boinga_translator import paths

CLIP_LEVEL = 0.999  # |sample| at or above this counts as clipped
# librosa.effects.trim: below peak - 40 dB counts as silence. Its centred 2048-sample frames
# make edge silence come out short by up to about one half frame (~70 ms at 16 kHz) for
# sharp onsets; a slow fade can come out slightly long.
TRIM_TOP_DB = 40
RMS_FLOOR_DB = -120.0  # rms_db of digital silence (log of 0 is undefined)
FACTS_FILE = "pretrained_facts.json"
TEXT_KEYS = ("n_chars", "n_words", "empty", "chars_not_in_model", "oov_ratio", "has_latin",
             "has_digits")
AUDIO_KEYS = ("decodable", "error", "sample_rate", "channels", "subtype", "duration_s", "peak",
              "clip_ratio", "rms_db", "lead_silence_s", "trail_silence_s")


def _edge_silence(mono: np.ndarray, sr: int, trim: Any) -> tuple[float, float]:
    """Leading/trailing silence in seconds; a clip with no signal is silence end to end."""
    if mono.size == 0:
        return 0.0, 0.0
    if not np.any(mono):  # trim measures against the clip's own peak, so it trims nothing here
        whole = mono.size / sr
        return whole, whole
    _, (start, end) = trim(mono, top_db=TRIM_TOP_DB)
    return float(start / sr), float((mono.size - end) / sr)


def audio_stats(wav_path: Path | str) -> dict[str, Any]:
    """Stats for one clip; an unreadable file gives decodable False + error, never raises.

    A missing librosa is an environment fault, not a bad clip, so that ImportError is raised.
    """
    from librosa.effects import trim  # heavy (numba): imported on first use, outside the try

    out: dict[str, Any] = dict.fromkeys(AUDIO_KEYS)
    try:
        info = sf.info(str(wav_path))
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop a corpus-wide audit
        out.update(decodable=False, error=f"{type(exc).__name__}: {exc}")
        return out
    out.update(decodable=True, sample_rate=int(sr), channels=int(info.channels),
               subtype=info.subtype, duration_s=data.shape[0] / sr)
    if not np.isfinite(data).all():  # only float wavs can hold NaN/inf
        out["error"] = "non-finite samples (NaN/inf)"
        return out
    try:
        mono = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
        lead, trail = _edge_silence(mono, sr, trim)
        out.update(
            peak=float(np.abs(data).max()) if data.size else 0.0,
            clip_ratio=float(np.mean(np.abs(data) >= CLIP_LEVEL)) if data.size else 0.0,
            rms_db=float(max(RMS_FLOOR_DB, 20 * np.log10(rms))) if rms > 0 else RMS_FLOOR_DB,
            lead_silence_s=lead, trail_silence_s=trail,
        )
    except Exception as exc:  # noqa: BLE001 - decoded but stats failed: report, keep going
        out["error"] = f"stats failed: {type(exc).__name__}: {exc}"
    return out


# --- text ------------------------------------------------------------------------------------


def load_model_charset(voice: str, facts_path: Path | None = None) -> frozenset[str]:
    """Code points the pretrained voice knows: characters | punctuations | {" "}.

    Not included: pad/bos/eos/blank specials (the eos symbol is the danda "।").
    """
    facts = json.loads((facts_path or paths.configs_dir() / FACTS_FILE).read_text(
        encoding="utf-8"))
    if voice not in facts:
        raise ValueError(f"no voice {voice!r} in pretrained facts; have {sorted(facts)}")
    return frozenset(facts[voice]["characters"]) | frozenset(facts[voice]["punctuations"]) | {" "}


def _is_digit(c: str) -> bool:
    return "0" <= c <= "9" or "০" <= c <= "৯"  # ASCII or Bengali digits ০-৯


def text_stats(text: str, charset: frozenset[str] | set[str]) -> dict[str, Any]:
    """Length and out-of-charset stats for one transcript, measured exactly as given."""
    visible = [c for c in text if not c.isspace()]
    missing = [c for c in visible if c not in charset]
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "empty": not visible,
        "chars_not_in_model": sorted(set(missing)),
        "oov_ratio": len(missing) / len(visible) if visible else 0.0,
        "has_latin": any("LATIN" in unicodedata.name(c, "").split() for c in visible),
        "has_digits": any(_is_digit(c) for c in visible),
    }
