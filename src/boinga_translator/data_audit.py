"""Corpus audit: per-clip audio stats and per-transcript text/charset stats for exclusions."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import unicodedata
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from boinga_translator import paths, text_frontend

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


# --- full-corpus audit -----------------------------------------------------------------------
# Each batch of results is its own part file (written to .tmp, then os.replace), so a dropped
# session leaves only whole parts behind; audit.csv is rebuilt from the parts atomically.

AUDIT_CSV = "audit.csv"
_KEY_COLS = ("clip_id", "wav_path", "ctg_text_nfc", "charset_id")  # stale if any changed
# Text stats are measured on ctg_text_model = text_frontend.prepare(ctg_text_nfc), the text
# the model reads (DECISIONS "Character map"); excluded_char marks rows dropped by the map.
_PART_COLS = (*_KEY_COLS, "ctg_text_model", *AUDIO_KEYS, *TEXT_KEYS, "chars_per_second",
              "excluded_char")
AUDIT_COLS = ("clip_id", "split_official", "wav_path", "ctg_text_nfc", "ctg_text_model",
              "charset_id", *AUDIO_KEYS, *TEXT_KEYS, "chars_per_second", "excluded_char",
              "duplicate_text")
_BOOL_COLS = ("decodable", "empty", "has_latin", "has_digits", "excluded_char", "duplicate_text")
_TEXT_COLS = ("clip_id", "split_official", "wav_path", "ctg_text_nfc", "ctg_text_model",
              "charset_id", "error", "subtype", "chars_not_in_model",
              "speaker_id")  # missing -> "" (no split/error)
_BATCH = 200  # rows per part file (a dropped session loses at most one unwritten batch)


def chars_per_second(text: str, duration: float | None, lead: float | None,
                     trail: float | None) -> float | None:
    """Non-space characters per second of non-silent audio; None if there is no speech."""
    if duration is None or lead is None or trail is None:
        return None
    speech = duration - lead - trail
    return sum(not c.isspace() for c in text) / speech if speech > 0 else None


def charset_id(voice: str, charset: frozenset[str],
               cmap: text_frontend.CharMap | None = None) -> str:
    """Voice + short hash of its charset and the char map: stats are valid only for both."""
    digest = hashlib.sha256("".join(sorted(charset)).encode("utf-8")).hexdigest()[:10]
    return f"{voice}-{digest}-{text_frontend.map_digest(cmap)}"


AuditJob = tuple[str, str, str, frozenset[str], str, text_frontend.CharMap]


def _audit_one(job: AuditJob) -> dict[str, Any]:
    """Pool worker: audio + text stats for one index row (module level so it pickles)."""
    clip_id, wav_path, text, charset, cs_id, cmap = job
    audio = audio_stats(wav_path) if wav_path else {**dict.fromkeys(AUDIO_KEYS),
                                                   "decodable": False, "error": "no wav file"}
    model_text = text_frontend.prepare(text, cmap)
    stats = text_stats(model_text, charset)
    stats["chars_not_in_model"] = "".join(stats["chars_not_in_model"])
    cps = chars_per_second(model_text, audio["duration_s"], audio["lead_silence_s"],
                           audio["trail_silence_s"])
    return {"clip_id": clip_id, "wav_path": wav_path, "ctg_text_nfc": text, "charset_id": cs_id,
            "ctg_text_model": model_text, **audio, **stats, "chars_per_second": cps,
            "excluded_char": text_frontend.excluded(text, cmap)}


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Same dtypes whether rows come fresh or from part files: text, bool, numeric."""
    df = df.copy()
    for col in df.columns:
        if col in _TEXT_COLS:
            df[col] = df[col].astype("string").fillna("").astype(str)  # all-NaN cols too
        elif col in _BOOL_COLS:
            df[col] = df[col].astype(str).eq("True")  # True / "True" -> True; blank -> False
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def read_audit(csv: Path) -> pd.DataFrame:
    """An audit CSV (audit.csv or a part) with the same dtypes audit() returns."""
    return _typed(pd.read_csv(csv, dtype=str, keep_default_na=False, encoding="utf-8"))


def _write_part(parts: Path, rows: list[dict[str, Any]]) -> None:
    """Write one batch as the next part-NNNNN.csv, via .tmp + os.replace (all or nothing)."""
    n = 1 + max((int(p.name[5:10]) for p in parts.glob("part-*.csv")), default=-1)
    tmp, final = parts / f"part-{n:05d}.csv.tmp", parts / f"part-{n:05d}.csv"
    pd.DataFrame(rows, columns=list(_PART_COLS)).to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, final)


def _read_parts(parts: Path) -> pd.DataFrame:
    """All completed parts (strict parse); later parts win for a repeated clip_id."""
    frames = []
    for p in sorted(parts.glob("part-*.csv")):
        try:
            frames.append(pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8"))
        except (ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"unreadable audit part {p} ({exc}); delete it and re-run") from exc
    if not frames:
        return pd.DataFrame(columns=list(_PART_COLS))
    # reindex: parts from an older audit lack newer columns; their old charset_id makes them
    # stale anyway, so missing columns only need to exist (as blanks) to be filtered out
    both = pd.concat(frames, ignore_index=True).reindex(columns=list(_PART_COLS))
    return _typed(both).drop_duplicates("clip_id", keep="last")


def audit(index_df: pd.DataFrame, voice: str, out_csv: Path | None = None,
          processes: int = 2) -> pd.DataFrame:
    """Audit every index row into out_csv (Drive), resuming from the part files next to it."""
    out_csv = out_csv or paths.drive_reports() / AUDIT_CSV
    parts = out_csv.with_name(out_csv.stem + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    charset = load_model_charset(voice)
    cmap = text_frontend.default_char_map()
    cs_id = charset_id(voice, charset, cmap)
    meta = index_df.assign(wav_path=index_df["wav_path"].fillna(""),
                           duplicate_text=index_df["ctg_text_nfc"].duplicated(keep=False))
    meta = meta.set_index("clip_id")
    done_ids = set(_current(_read_parts(parts), meta, cs_id)["clip_id"])
    print(f"skipped {len(done_ids)} rows already in {parts} ({cs_id})")
    jobs = [(cid, meta.at[cid, "wav_path"], meta.at[cid, "ctg_text_nfc"], charset, cs_id, cmap)
            for cid in meta.index if cid not in done_ids]
    batch: list[dict[str, Any]] = []
    with Pool(processes) if processes > 1 and jobs else contextlib.nullcontext() as pool:
        rows = (pool.imap_unordered(_audit_one, jobs, chunksize=16) if pool
                else map(_audit_one, jobs))
        for row in tqdm(rows, total=len(jobs), mininterval=10, desc="audit", disable=not jobs):
            batch.append(row)
            if len(batch) >= _BATCH:
                _write_part(parts, batch)
                batch = []
    if batch:
        _write_part(parts, batch)
    print(f"wrote {len(jobs)} rows to {parts}")
    return _assemble(meta, _current(_read_parts(parts), meta, cs_id), out_csv)


def _current(parts_df: pd.DataFrame, meta: pd.DataFrame, cs_id: str) -> pd.DataFrame:
    """Part rows still valid: same wav_path and text as the index, same charset."""
    now = parts_df.merge(meta[["wav_path", "ctg_text_nfc"]], left_on="clip_id",
                         right_index=True, suffixes=("", "_now"))
    keep = ((now["wav_path"] == now["wav_path_now"])
            & (now["ctg_text_nfc"] == now["ctg_text_nfc_now"]) & (now["charset_id"] == cs_id))
    return now.loc[keep, list(_PART_COLS)]


def _assemble(meta: pd.DataFrame, parts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """One row per index row in index order, written atomically to out_csv."""
    keyed = parts_df.set_index("clip_id")
    missing = [cid for cid in meta.index if cid not in keyed.index]
    if missing:
        raise RuntimeError(f"audit incomplete: {len(missing)} index rows have no result "
                           f"(e.g. {missing[:5]}); re-run to resume from {out_csv.parent}")
    result = keyed.loc[list(meta.index)].reset_index()
    result["split_official"] = meta["split_official"].to_numpy()
    result["duplicate_text"] = meta["duplicate_text"].to_numpy()
    result = _typed(result[list(AUDIT_COLS)])
    tmp = out_csv.with_name(out_csv.name + ".tmp")
    result.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, out_csv)
    print(f"PASS: audited {len(result)} clips")
    return result
