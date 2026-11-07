"""Corpus index: one row per spreadsheet line with its wav, dialect text, translation and split."""

# Layout (DECISIONS "Corpus layout" and "Corpus row <-> wav mapping"): the wav stem is the
# Excel row number of its line in Dataset/Chittagonian_translated.xlsx (header = row 1); the
# split is the folder under Dataset/Speech/. ctg_text is kept exactly as published.

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from boinga_translator import paths, text_frontend

XLSX_REL = Path("Dataset") / "Chittagonian_translated.xlsx"
SPEECH_REL = Path("Dataset") / "Speech"
SPLIT_FOLDERS = {"training": "train", "validation": "val", "test": "test"}
CTG_COL = "Chittagonian Dialect"
BN_COL = "Standard Bangla"
COLUMNS = ["clip_id", "wav_path", "ctg_text", "ctg_text_nfc", "bn_text", "split_official",
           "speaker_id", "wav_exists"]


def _raw_name(column: str) -> str:
    return "raw_" + re.sub(r"\W+", "_", column.strip()).strip("_").lower()


def _read_rows(xlsx: Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Header and (excel_row, cells as str) for every row of the first sheet."""
    import openpyxl  # in the [train] extra; imported here so the package imports without it

    wb = openpyxl.load_workbook(xlsx, read_only=True)
    try:
        rows = wb.worksheets[0].iter_rows(values_only=True)
        header = ["" if v is None else str(v) for v in next(rows)]
        body = [(r, ["" if v is None else str(v) for v in cells])
                for r, cells in enumerate(rows, start=2)]
    finally:
        wb.close()
    while body and not any(c.strip() for c in body[-1][1]):  # styled but empty trailing rows
        body.pop()
    for col in (CTG_COL, BN_COL):
        if col not in header:
            raise ValueError(f"{xlsx.name}: expected a {col!r} column, found {header}")
    return header, body


def _wavs(speech: Path) -> dict[int, tuple[Path, str]]:
    """wav stem -> (absolute path, official split); a stem in two splits is an error."""
    found: dict[int, tuple[Path, str]] = {}
    for folder, split in SPLIT_FOLDERS.items():
        if not (speech / folder).is_dir():
            raise FileNotFoundError(f"split folder missing (incomplete unpack?): {speech / folder}")
        for wav in sorted((speech / folder).glob("*.wav")):
            if not wav.stem.isdigit():
                raise ValueError(f"wav name is not an Excel row number: {wav}")
            stem = int(wav.stem)
            if stem in found:
                raise ValueError(f"wav {stem} is in more than one split: "
                                 f"{found[stem][0]} and {wav}")
            found[stem] = (wav.resolve(), split)
    return found


def load_index(root: Path | None = None) -> pd.DataFrame:
    """Index the unpacked corpus under root (default paths.data_fast())."""
    root = paths.data_fast() if root is None else Path(root)
    xlsx = root / XLSX_REL
    if not xlsx.is_file():
        raise FileNotFoundError(f"corpus workbook not found: {xlsx}")
    header, body = _read_rows(xlsx)
    wavs = _wavs(root / SPEECH_REL)
    orphans = sorted(set(wavs) - {r for r, _ in body})
    if orphans:
        raise ValueError(f"{len(orphans)} wav(s) have no spreadsheet row: {orphans[:10]}")
    ctg_at = header.index(CTG_COL)
    silent = [r for r, cells in body if r in wavs and not cells[ctg_at].strip()]
    if silent:
        raise ValueError(f"{len(silent)} row(s) with a wav have no {CTG_COL!r} text: {silent[:10]}")
    extra = [c for c in header if c not in (CTG_COL, BN_COL)]
    records = []
    for excel_row, cells in body:
        cell = dict(zip(header, cells, strict=False))
        wav, split = wavs.get(excel_row, (None, None))
        records.append({
            "clip_id": str(excel_row),
            "wav_path": str(wav) if wav else None,
            "ctg_text": cell.get(CTG_COL, ""),
            "ctg_text_nfc": text_frontend.normalise(cell.get(CTG_COL, "")),
            "bn_text": cell.get(BN_COL, ""),
            "split_official": split,
            "speaker_id": None,  # the corpus has no speaker field (DECISIONS "Corpus layout")
            "wav_exists": wav is not None,
            **{_raw_name(c): cell.get(c, "") for c in extra},
        })
    return pd.DataFrame(records, columns=[*COLUMNS, *(_raw_name(c) for c in extra)])


def describe(df: pd.DataFrame) -> list[str]:
    """Summary lines for `boinga index`: split and audio counts, speakers, 5 sample rows."""
    lines = ["rows by split_official:"]
    counts = df["split_official"].fillna("(none)").value_counts()
    lines += [f"  {split} {n}" for split, n in counts.items()]
    lines += [f"wav_exists {flag} {int((df['wav_exists'] == flag).sum())}" for flag in (True, False)]
    speakers = df["speaker_id"].dropna().unique()
    lines.append(f"speakers: {len(speakers)}" if len(speakers) else "speakers: no speaker field")
    lines.append("sample rows:")
    for row in df.head(5).itertuples():
        lines.append(f"  {row.clip_id} {row.split_official} {row.ctg_text_nfc[:60]}")
    return lines
