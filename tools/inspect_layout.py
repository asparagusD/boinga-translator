"""Describe the raw corpus tree (folders, tables, wav headers, metadata values) for the repo."""

# Usage: python tools/inspect_layout.py ROOT
# Writes reports/corpus_layout.txt (+ Drive copy), prints it between BEGIN/END markers, then
# "PASS: layout inspected". Unreadable files are reported, never fatal. Transcripts are shown
# verbatim (read-only, never normalised); long cells/lines are cut for display only.
# Each section has its own line budget and every line a character cap, so the report always
# keeps all sections and stays under report_io.MAX_EMIT_BYTES (Bengali = 3 bytes/char).

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

from boinga_translator import report_io

REPORT_REL = "reports/corpus_layout.txt"
TABLE_EXTS = (".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls")
MAX_TABLE_BYTES = 5 * 1024 * 1024
MAX_FILES_PER_EXT = 8  # table files shown per extension (a per-clip .txt layout has thousands)
MAX_SUBDIRS = 20  # subfolders listed per folder; the rest are summed into one line
BUDGET = {"tree": 100, "wavs": 6, "meta": 80, "tables": 200}  # lines per section (~400 total)
MAX_LINE_CHARS = 160
MAX_BYTES = 180_000
TREE_DEPTH = 3
HEAD_ROWS = 5
N_WAVS = 3
META_HINTS = ("speaker", "spk", "gender", "sex", "region", "district", "age", "split",
              "subset", "partition")


def _junk(path: Path) -> bool:
    """macOS zip leftovers: __MACOSX/ folders and ._* resource-fork files."""
    return path.name.startswith("._") or "__MACOSX" in path.parts


def _is_meta(column: str) -> bool:
    """A name token equal to a hint (Speaker_ID, age_group, SpeakerGender); not language/agent."""
    tokens = [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", column)]
    return any(t in META_HINTS or t.startswith(("speaker", "spk")) for t in tokens)


def _counts(files: list[Path]) -> str:
    c = Counter(p.suffix.lower() or "(none)" for p in files)
    return ", ".join(f"{ext} {n}" for ext, n in sorted(c.items())) or "no files"


def _files_in(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return [f for f in it if f.is_file() and not _junk(f)]


def _tree(folder: Path, depth: int, out: list[str]) -> None:
    subdirs = sorted(p for p in folder.iterdir() if p.is_dir() and not _junk(p))
    for i, sub in enumerate(subdirs):
        if i == MAX_SUBDIRS:
            rest = [f for d in subdirs[i:] for f in _files_in(d, True)]
            out.append(f"{'  ' * depth}... {len(subdirs) - i} more folders ({_counts(rest)})")
            break
        if depth < TREE_DEPTH:
            out.append(f"{'  ' * depth}{sub.name}/  ({_counts(_files_in(sub, False))})")
            _tree(sub, depth + 1, out)
        else:  # at the depth cap: count everything below, say so if there are subfolders
            deeper = ", incl. subfolders" if any(p.is_dir() for p in sub.iterdir()) else ""
            out.append(f"{'  ' * depth}{sub.name}/  ({_counts(_files_in(sub, True))}{deeper})")


def _read_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Sheet/file label -> DataFrame (all values as strings)."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):  # .xls needs xlrd (not installed): reported as an ERROR line
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        return {f" [{name}]": df.fillna("") for name, df in sheets.items()}
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):  # {key: record} or one record
            data = list(data.values()) if all(isinstance(v, dict) for v in data.values()) \
                else [data]
        return {"": pd.DataFrame(data).fillna("").astype(str)}
    kw = {"dtype": str, "keep_default_na": False, "encoding": "utf-8-sig"}
    df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", **kw)
    if ext == ".csv" and len(df.columns) == 1:  # maybe ';' / tab / '|'-separated
        sample = path.read_text(encoding="utf-8-sig")[:65536]
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=";\t|").delimiter
            return {f" (sep {sep!r} sniffed)": pd.read_csv(path, sep=sep, **kw)}
        except (csv.Error, pd.errors.ParserError):
            return {"": df}
    return {"": df}


def _txt_lines(path: Path, rel: str) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [f"{rel}: {len(lines)} lines", *(f"    {line}" for line in lines[:HEAD_ROWS])]


def _tables(root: Path, files: list[Path], meta: list[str]) -> list[str]:
    out: list[str] = []
    shown: Counter[str] = Counter()
    for path in (p for p in files if p.suffix.lower() in TABLE_EXTS):
        ext, rel = path.suffix.lower(), path.relative_to(root).as_posix()
        shown[ext] += 1
        if shown[ext] > MAX_FILES_PER_EXT:
            continue
        size = path.stat().st_size
        if size > MAX_TABLE_BYTES:
            out.append(f"{rel}: skipped ({size} B > {MAX_TABLE_BYTES} B)")
            continue
        try:
            if ext == ".txt":
                out.extend(_txt_lines(path, rel))
                continue
            for label, df in _read_tables(path).items():
                out.append(f"{rel}{label}: {len(df)} rows, {len(df.columns)} columns")
                out.append(f"  columns: {', '.join(map(str, df.columns))}")
                out.extend("    " + row for row in
                           df.head(HEAD_ROWS).to_string(max_colwidth=40).splitlines())
                for col in (c for c in df.columns if _is_meta(str(c))):
                    meta.append(f"{rel}{label} :: {col} ({df[col].nunique()} unique)")
                    meta.extend(f"    {v}: {n}" for v, n in df[col].value_counts().head(20).items())
        except Exception as exc:  # noqa: BLE001 - any parse error is reported, not fatal
            out.append(f"{rel}: ERROR {type(exc).__name__}: {exc}")
    out += [f"... {n - MAX_FILES_PER_EXT} more {ext} files not shown"
            for ext, n in sorted(shown.items()) if n > MAX_FILES_PER_EXT]
    return out


def _wavs(root: Path, files: list[Path]) -> list[str]:
    wavs = [p for p in files if p.suffix.lower() == ".wav"]
    out = [f"wav files: {len(wavs)}"]
    picks = sorted({0, len(wavs) // 2, len(wavs) - 1}) if wavs else []
    for path in [wavs[i] for i in picks][:N_WAVS]:
        rel = path.relative_to(root).as_posix()
        try:
            info = sf.info(str(path))
            out.append(f"{rel}: samplerate={info.samplerate} channels={info.channels} "
                       f"duration={info.duration:.3f}s format={info.format} "
                       f"subtype={info.subtype}")
        except Exception as exc:  # noqa: BLE001 - libsndfile errors vary
            out.append(f"{rel}: ERROR {type(exc).__name__}: {exc}")
    return out


def _section(title: str, lines: list[str], budget: int) -> list[str]:
    """Title + lines cut to MAX_LINE_CHARS each and to `budget` lines (with a marker)."""
    lines = [ln if len(ln) <= MAX_LINE_CHARS else ln[:MAX_LINE_CHARS - 1] + "…" for ln in lines]
    if len(lines) > budget:
        lines = [*lines[:budget - 1], f"... truncated ({len(lines) - budget + 1} more lines)"]
    return ["", f"== {title} ==", *lines]


def render(root: Path) -> str:
    """The whole layout report for root; every section kept, under MAX_BYTES of UTF-8."""
    every = sorted(p for p in root.rglob("*") if p.is_file())
    files = [p for p in every if not _junk(p)]
    tree = [f"./  ({_counts([f for f in files if f.parent == root])})"]
    _tree(root, 1, tree)
    meta: list[str] = []
    tables = _tables(root, files, meta)
    out = [f"Corpus layout of {root}",
           f"macOS zip leftovers skipped (__MACOSX/, ._*): {len(every) - len(files)} files",
           *_section("Tree (depth 3; files per extension)", tree, BUDGET["tree"]),
           *_section("Wav headers (first, middle, last)", _wavs(root, files), BUDGET["wavs"]),
           *_section("Metadata columns (speaker/gender/region/age/split; top 20 values)",
                     meta or ["none found"], BUDGET["meta"]),
           *_section(f"Tables (< {MAX_TABLE_BYTES // 2**20} MB; first {HEAD_ROWS} rows; "
                     "long cells cut for display only)", tables, BUDGET["tables"])]
    text = "\n".join(out) + "\n"
    while len(text.encode("utf-8")) > MAX_BYTES:  # tables come last, so they are cut first
        out = out[:-2] + ["... truncated (byte budget)"]
        text = "\n".join(out) + "\n"
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="unpacked corpus folder (paths.data_fast() on Colab)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not args.root.strip() or not root.is_dir():  # "" would silently mean "."
        print(f"FAIL: not a folder: {args.root!r}")
        return 1
    if not any(p.suffix.lower() == ".wav" and not _junk(p) for p in root.rglob("*")):
        print(f"FAIL: no .wav files under {root} (did the download cell pass?)")
        return 1
    repo_path, _drive_copy = report_io.write_report(render(root), REPORT_REL)
    report_io.emit(repo_path, REPORT_REL)
    print("PASS: layout inspected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
