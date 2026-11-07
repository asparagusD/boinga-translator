"""Training-data prep: the model's sample rate and the per-clip audio transform (mono, rate, trim, peak)."""

# dialect-data-rules: mono, resample to the checkpoint's rate (never change the model's
# rate), trim only leading/trailing silence keeping >= 100 ms, peak-safe normalisation.
# DECISIONS "Audio preprocessing" records the choices and their reasons.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from boinga_translator import data_audit, paths, speaker_select

PAD_S = 0.1  # silence kept before the first and after the last non-silent frame
TRIM_TOP_DB = data_audit.TRIM_TOP_DB  # same silence threshold as the audit (40 dB below peak)
PEAK_DBFS = -1.0
RES_TYPE = "soxr_hq"


def target_sample_rate(selection_path: Path | None = None, facts_path: Path | None = None) -> int:
    """The base voice's sample rate: configs/speaker_selection.json -> pretrained_facts.json."""
    sel = json.loads((selection_path or paths.REPO / speaker_select.SELECTION_REL).read_text(
        encoding="utf-8"))
    facts = json.loads((facts_path or paths.configs_dir() / data_audit.FACTS_FILE).read_text(
        encoding="utf-8"))
    voice = sel.get("base_voice")
    if voice not in facts:
        raise ValueError(f"base voice {voice!r} not in pretrained facts; have {sorted(facts)}")
    rate = facts[voice].get("sample_rate")
    if type(rate) is not int or rate <= 0:
        raise ValueError(f"pretrained facts give {voice!r} no valid sample_rate: {rate!r}")
    return rate


def transform_audio(in_wav: Path | str, out_wav: Path | str, target_sr: int) -> dict[str, Any]:
    """Write in_wav as a mono 16-bit wav at target_sr, edge silence trimmed, peak at -1 dBFS.

    Returns in_sr, out_sr, in_dur, out_dur (s), trimmed_s (edge silence cut, at target_sr),
    gain_db and upsampled (in_sr < target_sr). An empty, non-finite or silent clip raises
    ValueError and nothing is written; out_wav appears only when complete.
    """
    from librosa import resample  # heavy (numba): imported on first use
    from librosa.effects import trim

    data, in_sr = sf.read(str(in_wav), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError(f"{in_wav}: empty clip")
    if not np.isfinite(data).all():
        raise ValueError(f"{in_wav}: non-finite samples (NaN/inf)")
    mono = data.mean(axis=1)
    y = mono if in_sr == target_sr else resample(
        mono, orig_sr=in_sr, target_sr=target_sr, res_type=RES_TYPE)
    if not np.any(y):
        raise ValueError(f"{in_wav}: silent clip (no non-zero sample)")
    _, (start, end) = trim(y, top_db=TRIM_TOP_DB)
    pad = round(PAD_S * target_sr)
    kept = y[max(0, start - pad):min(y.size, end + pad)]
    gain = 10 ** (PEAK_DBFS / 20) / float(np.abs(kept).max())
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        sf.write(str(tmp), kept * gain, target_sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)  # no half-written file for a later "skip done" check
        raise
    return {"in_sr": int(in_sr), "out_sr": int(target_sr), "in_dur": mono.size / in_sr,
            "out_dur": kept.size / target_sr, "trimmed_s": (y.size - kept.size) / target_sr,
            "gain_db": float(20 * np.log10(gain)), "upsampled": in_sr < target_sr}
