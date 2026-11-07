"""Tests for tools/inspect_layout.py: tree counts, table heads, wav headers, metadata values, cap."""

from __future__ import annotations

import json
from pathlib import Path

import inspect_layout
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import paths, report_io

CTG = "আঁই বাজারত যাইয়ুম"  # য় decomposed (য + nukta): must survive verbatim


def _wav(path: Path, seconds: float = 0.25, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(sr * seconds), dtype="float32"), sr)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    return repo


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    for split, n in (("train", 4), ("test", 2)):
        for i in range(n):
            _wav(root / "Dataset" / "audio" / split / f"{split}_{i:03d}.wav")
    meta = pd.DataFrame({
        "file_name": ["train_000.wav", "train_001.wav", "test_000.wav"],
        "chittagonian": [CTG, CTG, "তুই কডে যাছ"],
        "standard_bangla": ["আমি বাজারে যাব", "আমি বাজারে যাব", "তুই কোথায় যাস"],
        "Speaker_ID": ["S1", "S2", "S1"],
        "split": ["train", "train", "test"],
    })
    meta.to_csv(root / "Dataset" / "metadata.csv", index=False)
    meta.to_csv(root / "Dataset" / "metadata.tsv", sep="\t", index=False)
    (root / "Dataset" / "README.txt").write_text("line one\nline two\n", encoding="utf-8")
    (root / "Dataset" / "info.json").write_text(
        json.dumps([{"gender": "female", "age": 30}, {"gender": "male", "age": 41}]),
        encoding="utf-8")
    deep = root / "Dataset" / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "deep.wav").write_bytes(b"not really audio")
    return root


def test_render_tree_tables_wavs_and_meta(corpus: Path) -> None:
    text = inspect_layout.render(corpus)
    assert "Dataset/  (.csv 1, .json 1, .tsv 1, .txt 1)" in text
    assert "    train/  (.wav 4)" in text
    assert "b/  (.wav 1, incl. subfolders)" in text  # Dataset=1, a=2, b=3
    assert "c/  (" not in text  # depth cap 3
    assert "Dataset/metadata.csv: 3 rows, 5 columns" in text
    assert "columns: file_name, chittagonian, standard_bangla, Speaker_ID, split" in text
    assert CTG in text  # transcripts shown verbatim
    assert "Dataset/README.txt: 2 lines" in text
    assert "Dataset/info.json: 2 rows, 2 columns" in text
    assert "wav files: 7" in text
    assert "samplerate=16000 channels=1" in text
    assert "duration=0.250s" in text
    assert "Speaker_ID" in text and "S1: 2" in text
    assert "gender" in text and "female: 1" in text
    assert "split" in text and "train: 2" in text
    meta_section = text.split("== Metadata")[1].split("== Tables")[0]
    assert "chittagonian" not in meta_section  # not a metadata column
    assert "macOS zip leftovers skipped (__MACOSX/, ._*): 0 files" in text


def test_macos_zip_leftovers_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "c"
    _wav(root / "audio" / "a.wav")
    (root / "audio" / "._a.wav").write_bytes(b"\x00\x05\x16\x07")
    (root / "__MACOSX" / "audio").mkdir(parents=True)
    (root / "__MACOSX" / "audio" / "._a.wav").write_bytes(b"\x00")
    text = inspect_layout.render(root)
    assert "wav files: 1" in text
    assert "__MACOSX/  (" not in text
    assert "macOS zip leftovers skipped (__MACOSX/, ._*): 2 files" in text
    assert "ERROR" not in text


def test_semicolon_csv_separator_is_sniffed(tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "m.csv").write_bytes("id;text;x\n1;হ;a\n2;ক;b\n".encode())
    (root / "one.csv").write_bytes(b"id\n1\n2\n")
    text = inspect_layout.render(root)
    assert "m.csv (sep ';' sniffed): 2 rows, 3 columns" in text
    assert "one.csv: 2 rows, 1 columns" in text


def test_wide_bengali_tables_stay_under_emit_limit(tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    cols = {f"কলাম_{i:02d}_" + "অ" * 30: ["চট্টগ্রামের ভাষা " * 8] * 6 for i in range(12)}
    for i in range(8):
        pd.DataFrame(cols).to_csv(root / f"t{i}.csv", index=False)
        pd.DataFrame(cols).to_csv(root / f"t{i}.tsv", sep="\t", index=False)
        (root / f"t{i}.txt").write_text(("অ" * 500 + "\n") * 6, encoding="utf-8")
    text = inspect_layout.render(root)
    assert len(text.encode("utf-8")) <= inspect_layout.MAX_BYTES < report_io.MAX_EMIT_BYTES
    assert max(len(line) for line in text.splitlines()) <= inspect_layout.MAX_LINE_CHARS
    assert "== Tables" in text


@pytest.mark.parametrize("name, meta", [
    ("Speaker_ID", True), ("speaker", True), ("SpeakerGender", True), ("age_group", True),
    ("Region", True), ("split", True), ("spk", True), ("spkid", True), ("SpkID", True), ("language", False), ("stage", False),
    ("file_name", False), ("chittagonian", False), ("Agent", False), ("agenda", False),
])
def test_metadata_column_names(name: str, meta: bool) -> None:
    assert inspect_layout._is_meta(name) is meta


def test_xlsx_sheets_are_read(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    root = tmp_path / "c"
    root.mkdir()
    pd.DataFrame({"region": ["Ctg", "Ctg", "Cox"]}).to_excel(root / "m.xlsx", index=False)
    text = inspect_layout.render(root)
    assert "m.xlsx [Sheet1]: 3 rows, 1 columns" in text
    assert "Ctg: 2" in text


def test_big_and_unreadable_files_do_not_stop_the_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "big.csv").write_bytes(b"a\n1\n2\n")
    (root / "bad.json").write_bytes(b"{x")
    (root / "broken.wav").write_bytes(b"nope")
    monkeypatch.setattr(inspect_layout, "MAX_TABLE_BYTES", 5)
    text = inspect_layout.render(root)
    assert "big.csv: skipped (6 B > 5 B)" in text
    assert "bad.json: ERROR" in text
    assert "broken.wav: ERROR" in text


def test_many_files_of_one_type_are_capped(tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    for i in range(inspect_layout.MAX_FILES_PER_EXT + 3):
        (root / f"t{i:02d}.txt").write_text("x\n", encoding="utf-8")
    text = inspect_layout.render(root)
    assert "... 3 more .txt files not shown" in text


def test_deep_tree_is_capped_but_keeps_every_section(tmp_path: Path) -> None:
    root = tmp_path / "c"
    for s in range(25):
        for k in range(25):
            (root / f"spk{s:02d}" / f"sess{k:02d}").mkdir(parents=True)
    _wav(root / "spk00" / "sess00" / "a.wav")
    pd.DataFrame({"speaker": ["S1"]}).to_csv(root / "meta.csv", index=False)
    text = inspect_layout.render(root)
    tree = text.split("== Tree")[1].split("== Wav")[0].strip().splitlines()
    assert len(tree) == inspect_layout.BUDGET["tree"] + 1  # title line remainder + budget
    assert tree[-1].startswith("... truncated (")
    assert "samplerate=16000" in text
    assert "meta.csv :: speaker (1 unique)" in text
    assert "meta.csv: 1 rows, 1 columns" in text


def test_many_subfolders_are_summarised(tmp_path: Path) -> None:
    root = tmp_path / "c"
    for i in range(inspect_layout.MAX_SUBDIRS + 4):
        _wav(root / f"spk{i:02d}" / "a.wav")
    text = inspect_layout.render(root)
    assert "... 4 more folders (.wav 4)" in text


def test_main_writes_and_emits_report(
        sandbox: Path, corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert inspect_layout.main([str(corpus)]) == 0
    out = capsys.readouterr().out
    report = sandbox / "reports" / "corpus_layout.txt"
    assert report.read_text(encoding="utf-8").startswith("Corpus layout of ")
    assert "=== BEGIN reports/corpus_layout.txt ===" in out
    assert out.rstrip().endswith("PASS: layout inspected")


def test_main_fails_on_empty_or_wavless_tree(
        sandbox: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """C3 run 1: C3.1 failed, C3.2 then printed PASS on an empty data_fast()."""
    root = tmp_path / "empty"
    root.mkdir()
    assert inspect_layout.main([str(root)]) == 1
    assert "FAIL: no .wav files under" in capsys.readouterr().out
    (root / "only.csv").write_bytes(b"a\n1\n")
    assert inspect_layout.main([str(root)]) == 1
    out = capsys.readouterr().out
    assert "FAIL: no .wav files under" in out and "PASS" not in out
    assert not (sandbox / "reports" / "corpus_layout.txt").exists()


def test_main_fails_on_missing_root(
        sandbox: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert inspect_layout.main([str(tmp_path / "nope")]) == 1
    assert "FAIL: not a folder" in capsys.readouterr().out
    assert inspect_layout.main([""]) == 1  # a failed $(python -c ...) must not inspect "."
    assert "FAIL: not a folder: ''" in capsys.readouterr().out
