"""Audit summary: exclusion rules (clean_mask), reports/audit_summary.md, clean clips to hear."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from boinga_translator import corpus, data_audit, paths, report_io, settings

SUMMARY_REL = "reports/audit_summary.md"
MIN_S, MAX_S = 1.0, 15.0
MAX_CLIP_RATIO, MIN_RMS_DB = 0.001, -45.0
CPS_PCT = (1, 99)
# reason -> (label in the report, predicate on the audit frame). A clip can have several.
RULES: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "undecodable": ("undecodable (or no wav)", lambda d: ~d["decodable"] | (d["error"] != "")),
    "too_short": (f"too short (< {MIN_S:g} s)", lambda d: d["duration_s"] < MIN_S),
    "too_long": (f"too long (> {MAX_S:g} s)", lambda d: d["duration_s"] > MAX_S),
    "clipped": (f"clipped (clip_ratio > {MAX_CLIP_RATIO:g})",
                lambda d: d["clip_ratio"] > MAX_CLIP_RATIO),
    "quiet": (f"quiet (rms_db < {MIN_RMS_DB:g})", lambda d: d["rms_db"] < MIN_RMS_DB),
    "empty_text": ("empty text", lambda d: d["empty"]),
    # DECISIONS "Character map": judged on the mapped text; any unknown char would be dropped.
    "excluded_char": ("excluded character (char_map: / : * digits)", lambda d: d["excluded_char"]),
    "oov": ("unknown to the model after char_map", lambda d: d["chars_not_in_model"] != ""),
    "speech_rate": ("speech rate (chars/s outside 1st-99th percentile, or no speech)",
                    lambda d: _cps_outlier(d)),
    "duplicate_text": ("duplicate text", lambda d: d["duplicate_text"]),
}


def _cps_outlier(df: pd.DataFrame) -> pd.Series:
    cps = df["chars_per_second"]
    lo, hi = cps.dropna().quantile([CPS_PCT[0] / 100, CPS_PCT[1] / 100])
    return cps.isna() | (cps < lo) | (cps > hi)


def exclusion_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """One bool column per exclusion reason (RULES order)."""
    missing = [c for c in ("excluded_char", "ctg_text_model") if c not in df]
    if missing:
        raise ValueError(f"audit table lacks {missing} (made before the char map): "
                         "re-run `boinga audit`")
    return pd.DataFrame({name: rule(df).fillna(False).astype(bool)
                         for name, (_, rule) in RULES.items()}, index=df.index)


def clean_mask(df: pd.DataFrame) -> pd.Series:
    """True for clips no exclusion rule flags: the training candidates."""
    return ~exclusion_reasons(df).any(axis=1)


def _hours(seconds: pd.Series) -> str:
    return f"{seconds.fillna(0).sum() / 3600:.2f}"


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    return [f"| {' | '.join(header)} |", f"|{'---|' * len(header)}",
            *(f"| {' | '.join(map(cell, r))} |" for r in rows)]


def render(df: pd.DataFrame) -> str:
    """The markdown summary of one audit table."""
    if df.empty:
        raise ValueError("audit table has no rows: run `boinga audit` first")
    reasons, clean = exclusion_reasons(df), clean_mask(df)
    dur = df["duration_s"]
    split = df["split_official"].replace("", "(none)")
    lines = ["# Corpus audit summary", "",
             (f"Charset: {', '.join(sorted(set(df['charset_id'])))}. "
              f"Rows: {len(df)}; decodable clips: {int(df['decodable'].sum())}, "
              f"{_hours(dur)} h."), "", "## Per split", ""]
    lines += _table(["split", "clips", "hours", "clean clips", "clean hours"], [
        [s, int((split == s).sum()), _hours(dur[split == s]), int((clean & (split == s)).sum()),
         _hours(dur[clean & (split == s)])] for s in sorted(split.unique())])
    fmt = df[df["decodable"]].groupby(["sample_rate", "channels", "subtype"]).size()
    lines += ["", "## Audio formats", ""]
    lines += _table(["sample_rate", "channels", "subtype", "clips"],
                    [[int(sr), int(ch), st, n] for (sr, ch, st), n in fmt.items()])
    lines += ["", "## Exclusions (a clip can have several reasons)", ""]
    lines += _table(["reason", "clips", "hours"], [
        [RULES[r][0], int(reasons[r].sum()), _hours(dur[reasons[r]])] for r in RULES])
    lines += ["", f"Clean: {int(clean.sum())} clips, {_hours(dur[clean])} h.", "",
              "## Speakers", ""]
    # audit.csv has no speaker_id today (the corpus has none); kept for a corpus that does.
    if "speaker_id" in df and (df["speaker_id"] != "").any():
        mins = (dur[clean].groupby(df["speaker_id"][clean]).sum() / 60).sort_values(
            ascending=False)
        lines += _table(["speaker", "clean minutes"], [[s, f"{m:.1f}"] for s, m in mins.items()])
    else:
        lines.append("no speaker field in the corpus (DECISIONS \"Corpus layout\")")
    counts = Counter(c for chars in df["chars_not_in_model"] for c in chars)
    lines += ["", "## Characters not in the model after char_map (top 20, rows containing)", ""]
    lines += _table(["char", "code", "name", "rows"], [
        [c, f"U+{ord(c):04X}", unicodedata.name(c, "?"), n] for c, n in counts.most_common(20)])
    return "\n".join(lines) + "\n"


def summarise(audit_csv: Path | None = None) -> str:
    """Write reports/audit_summary.md (repo + Drive copy), emit it, return its text."""
    df = data_audit.read_audit(audit_csv or paths.drive_reports() / data_audit.AUDIT_CSV)
    text = render(df)
    repo_path, _drive = report_io.write_report(text, SUMMARY_REL)
    report_io.emit(repo_path, SUMMARY_REL)
    print("PASS: summary written")
    return text


def sample_clips(n: int = 20, seed: int = settings.SEED, audit_csv: Path | None = None,
                 index_df: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Seeded random clean clips as (wav paths, labels with the dialect text and translation)."""
    df = data_audit.read_audit(audit_csv or paths.drive_reports() / data_audit.AUDIT_CSV)
    picked = df[clean_mask(df)]
    picked = picked.sample(min(n, len(picked)), random_state=seed)
    index = (corpus.load_index() if index_df is None else index_df).set_index("clip_id")
    labels = [f"{r.clip_id} [{r.split_official}] CTG: {index.at[r.clip_id, 'ctg_text']} | "
              f"BN: {index.at[r.clip_id, 'bn_text']}" for r in picked.itertuples()]
    return list(picked["wav_path"]), labels
