"""Training-data prep: model sample rate, per-clip audio transform, leak-free train/val/test split."""

# dialect-data-rules: mono, resample to the checkpoint's rate (never change the model's
# rate), trim only leading/trailing silence keeping >= 100 ms, peak-safe normalisation.
# DECISIONS "Audio preprocessing" records the choices and their reasons.

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from boinga_translator import data_audit, paths, settings, speaker_select

PAD_S = 0.1  # silence kept before the first and after the last non-silent frame
TRIM_TOP_DB = data_audit.TRIM_TOP_DB  # same silence threshold as the audit (40 dB below peak)
PEAK_DBFS = -1.0
RES_TYPE = "soxr_hq"
HELD_OUT_FRACTION = 0.05  # test (if sampled) and val, each
MIN_HELD_OUT = 20  # clips; also the least official-test clips that count as a usable test set
SPLIT_NAMES = ("train", "val", "test")
SPLITS_SUMMARY_REL = "configs/splits_summary.json"


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


# --- splits ----------------------------------------------------------------------------------


def _held_out(n: int) -> int:
    return max(MIN_HELD_OUT, round(HELD_OUT_FRACTION * n))


def _sample(ids: pd.Series, n: int, rng: np.random.Generator) -> set[str]:
    """n ids drawn without replacement; the input order does not matter (sorted first)."""
    pool = sorted(ids)
    return {pool[i] for i in rng.permutation(len(pool))[:n]}


def make_splits(index_df: pd.DataFrame, selection_ids: Iterable[str],
                seed: int = settings.SEED) -> pd.DataFrame:
    """One row per kept clip: clip_id, split (train/val/test), duration_s.

    index_df needs clip_id, ctg_text_nfc, split_official and duration_s. Only selected clips
    are kept; clips sharing a ctg_text_nfc keep the longest one (ties: lowest clip_id), so no
    transcript reaches two splits. Test = the official test clips if >= MIN_HELD_OUT remain,
    else a seeded HELD_OUT_FRACTION sample (min MIN_HELD_OUT); val = a seeded
    HELD_OUT_FRACTION (min MIN_HELD_OUT) of the rest; train = what is left. attrs carry
    seed, test_source, n_selected and dropped_duplicate_text for save_splits.
    """
    need = {"clip_id", "ctg_text_nfc", "split_official", "duration_s"}
    if missing := need - set(index_df.columns):
        raise ValueError(f"index_df lacks column(s) {sorted(missing)}")
    wanted = pd.Index(pd.unique(pd.Series(list(selection_ids), dtype=str)))
    df = index_df.assign(clip_id=index_df["clip_id"].astype(str)).set_index("clip_id")
    if unknown := sorted(set(wanted) - set(df.index)):
        raise ValueError(f"{len(unknown)} selected clip(s) not in the index: {unknown[:5]}")
    df = df.loc[wanted].reset_index()
    order = df["clip_id"].str.zfill(12)
    df = (df.assign(_o=order).sort_values(["duration_s", "_o"], ascending=[False, True])
          .drop_duplicates("ctg_text_nfc").sort_values("_o"))
    rng = np.random.default_rng(seed)
    official = df.loc[df["split_official"] == "test", "clip_id"]
    if len(official) >= MIN_HELD_OUT:
        test, source = set(official), "official"
    else:
        test, source = _sample(df["clip_id"], _held_out(len(df)), rng), "sampled"
    rest = df.loc[~df["clip_id"].isin(test), "clip_id"]
    val = _sample(rest, _held_out(len(rest)), rng)
    if len(rest) <= len(val):
        raise ValueError(f"no clips left for train: {len(df)} unique texts, "
                         f"{len(test)} test, {len(val)} val")
    split = np.where(df["clip_id"].isin(test), "test",
                     np.where(df["clip_id"].isin(val), "val", "train"))
    out = pd.DataFrame({"clip_id": df["clip_id"].to_numpy(), "split": split,
                        "duration_s": df["duration_s"].astype(float).to_numpy()})
    by_text = pd.Series(split, index=df["ctg_text_nfc"].to_numpy()).groupby(level=0).nunique()
    if (by_text > 1).any():  # not `assert`: the guarantee must hold under python -O too
        raise RuntimeError(f"transcript(s) in two splits: {list(by_text[by_text > 1].index[:3])}")
    out.attrs.update(seed=seed, test_source=source, n_selected=len(wanted),
                     dropped_duplicate_text=len(wanted) - len(df))
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_splits(df: pd.DataFrame, csv_path: Path | None = None,
                summary_path: Path | None = None) -> dict[str, Any]:
    """Write splits.csv (Drive, paths.splits_csv()) and configs/splits_summary.json.

    Refuses to replace a saved splits.csv whose test clips differ (the test set is fixed
    once and never trained on). Returns the summary.
    """
    if not {"seed", "test_source", "dropped_duplicate_text"} <= set(df.attrs):
        raise ValueError("save_splits needs the frame make_splits returns (attrs missing)")
    csv_path = csv_path or paths.splits_csv()
    summary_path = summary_path or paths.REPO / SPLITS_SUMMARY_REL
    new_test = set(df.loc[df["split"] == "test", "clip_id"].astype(str))
    if csv_path.is_file():
        old = pd.read_csv(csv_path, dtype={"clip_id": str})
        if set(old.loc[old["split"] == "test", "clip_id"]) != new_test:
            raise ValueError(f"{csv_path} already fixes a different test set; refusing to "
                             "change it (delete it on purpose to re-split)")
    data = df[["clip_id", "split", "duration_s"]].to_csv(index=False, lineterminator="\n")
    blob = data.encode("utf-8")
    _atomic_write(csv_path, blob)
    counts = df["split"].value_counts()
    minutes = df.groupby("split")["duration_s"].sum() / 60
    summary = {
        "counts": {k: int(counts.get(k, 0)) for k in SPLIT_NAMES},
        "minutes": {k: round(float(minutes.get(k, 0.0)), 2) for k in SPLIT_NAMES},
        "csv_sha256": hashlib.sha256(blob).hexdigest(),
        "seed": int(df.attrs["seed"]),
        "test_source": df.attrs["test_source"],
        "dropped_duplicate_text": int(df.attrs["dropped_duplicate_text"]),
    }
    _atomic_write(summary_path, (json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return summary
