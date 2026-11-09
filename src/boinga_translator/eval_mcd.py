"""Objective check for C10: mel-cepstral distortion (DTW) of each voice vs the real recordings."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from boinga_translator import paths, report_io
from boinga_translator.evaluate import EVAL_REL, REF_SET, VOICE_SETS, final_dir

N_MCC = 13  # coefficients 1-13; c0 (energy) is left out, so loudness does not count
TRIM_DB = 40  # leading/trailing silence trimmed before alignment
SECTION = "## Objective check (relative only)"


def _load(path: Path, rate: int | None = None) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if rate is not None and sr != rate:
        wav, sr = librosa.resample(wav, orig_sr=sr, target_sr=rate), rate
    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_DB)
    return (trimmed if trimmed.size else wav), sr


def _mcc(wav: np.ndarray, rate: int) -> np.ndarray:
    return librosa.feature.mfcc(y=wav, sr=rate, n_mfcc=N_MCC + 1)[1:]


def mcd_dtw(a_wav: Path, r_wav: Path) -> float:
    """MCD in dB of a_wav against r_wav after DTW alignment (r_wav's sample rate).

    sqrt(2) x the Euclidean distance of librosa MFCC frames (an orthonormal DCT of dB mel
    power), averaged along the DTW path. That is the textbook formula on librosa's power
    cepstrum; the usual SPTK mel-cepstrum is amplitude-based with other filterbank scaling,
    so values are about twice published MCDs and only comparable within this table.
    """
    ref, rate = _load(Path(r_wav))
    syn, _ = _load(Path(a_wav), rate)
    x, y = _mcc(syn, rate), _mcc(ref, rate)
    _, path = librosa.sequence.dtw(X=x, Y=y, metric="euclidean")
    dist = np.linalg.norm(x[:, path[:, 0]] - y[:, path[:, 1]], axis=0)
    return float(np.sqrt(2) * dist.mean())


def _section(table: pd.DataFrame) -> str:
    a, b = (table[s] for s in VOICE_SETS)
    lines = [SECTION, "",
             (f"MCD in dB (mel-cepstral distortion: librosa MFCC 1-{N_MCC}, silence trimmed, "
              "DTW-aligned; librosa's convention, not comparable to published MCD values) of "
              f"each voice against the real recording of the same prompt, "
              f"{len(table)} prompts; lower is closer to the real recording. The real speaker "
              "is not the voice of A or B, so MCD mixes speaker and dialect: only the A vs B "
              "difference means anything, and the listening test above outranks it."), "",
             "| prompt | A_baseline vs R | B_boinga vs R |", "|---|---|---|"]
    lines += [f"| {r.prompt_id} | {r.A_baseline:.2f} | {r.B_boinga:.2f} |"
              for r in table.itertuples()]
    lines += [f"| mean ± sd | {a.mean():.2f} ± {a.std():.2f} | {b.mean():.2f} ± {b.std():.2f} |",
              "", (f"B_boinga closer than A_baseline on {int((b < a).sum())} of {len(table)} "
                   "prompts.")]
    return "\n".join(lines) + "\n"


def mcd_table() -> pd.DataFrame:
    """MCD of A and B vs R for every prompt with a reference; (re)writes eval.md's section.

    Needs reports/eval.md from `boinga eval-score`; a re-run replaces the section. Emits
    eval.md for the repo. Returns one row per prompt: prompt_id, A_baseline, B_boinga.
    """
    report = paths.REPO / EVAL_REL
    if not report.is_file():
        raise FileNotFoundError(f"{EVAL_REL} not found: run `boinga eval-score` first")
    index = pd.read_csv(final_dir() / "index.csv", dtype=str)
    wav = {(r.id, r.set): Path(r.path) for r in index.itertuples()}
    ids = sorted({pid for pid, s in wav if s == REF_SET})
    rows = []
    missing = [f"{pid} {s}" for pid in ids for s in VOICE_SETS if (pid, s) not in wav]
    if missing:
        raise KeyError(f"final/index.csv has R_real but no clip for: {', '.join(missing)}")
    for pid in tqdm(ids, desc="MCD vs R", unit="prompt"):
        rows.append({"prompt_id": pid, **{s: mcd_dtw(wav[(pid, s)], wav[(pid, REF_SET)])
                                          for s in VOICE_SETS}})
    table = pd.DataFrame(rows, columns=["prompt_id", *VOICE_SETS])
    if table.empty:
        raise ValueError("no prompt has a real recording (R_real) in final/index.csv")
    text = report.read_text(encoding="utf-8")
    start = text.find(f"\n{SECTION}\n")
    if start < 0:  # first run: append at the end
        head, tail = text, ""
    else:  # re-run: replace only this section, keep any later "## " section
        end = text.find("\n## ", start + len(SECTION) + 2)
        head, tail = text[:start], ("" if end < 0 else text[end:])
    text = head.rstrip("\n") + "\n\n" + _section(table) + tail
    repo_path, _ = report_io.write_report(text, EVAL_REL)
    report_io.emit(repo_path, EVAL_REL)
    return table
