"""Tests for corpus.py and `boinga index`: row<->wav mapping, splits, verbatim text, errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import cli, corpus, paths

openpyxl = pytest.importorskip("openpyxl")

YA = "য়"  # precomposed য় (NFC decomposes it: U+09DF is composition-excluded)
ROWS = [  # Excel rows 2..5 -> wav stems 2..5
    ("আমি বাজারে যাব", "I will go to the market", f"আঁই  বাজারত যাই{YA}ুম "),
    ("তুমি কোথায়", "Where are you", "তুই কডে"),
    ("ভালো", "Good", "\tভালা\n"),
    ("অনেক লম্বা বাক্য", "A very long sentence", "ম্যালা লম্বা কথা"),
]


def _wav(path: Path) -> None:
    """0.1 s silence + 0.5 s of a 220 Hz tone at 16 kHz (audit tests need some 'speech')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)
    sf.write(path, np.concatenate([np.zeros(1600), tone]).astype("float32"), 16000)


def _corpus(root: Path, wavs: dict[str, list[int]], rows: list[tuple[str, ...]] = ROWS) -> Path:
    ds = root / "Dataset"
    ds.mkdir(parents=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Chittagonian_translated"
    ws.append(["Standard Bangla", "Translated English", "Chittagonian Dialect"])
    for row in rows:
        ws.append(list(row))
    wb.save(ds / "Chittagonian_translated.xlsx")
    for folder in ("training", "validation", "test"):
        (ds / "Speech" / folder).mkdir(parents=True)
    for folder, stems in wavs.items():
        for stem in stems:
            _wav(ds / "Speech" / folder / f"{stem}.wav")
    return root


@pytest.fixture
def root(tmp_path: Path) -> Path:
    # Excel row 5 has no wav (like the 5 long sentences in the real corpus)
    return _corpus(tmp_path / "c", {"training": [2, 3], "validation": [4], "test": []})


def test_columns_and_row_to_wav_mapping(root: Path) -> None:
    df = corpus.load_index(root)
    assert list(df.columns) == ["clip_id", "wav_path", "ctg_text", "ctg_text_nfc", "bn_text",
                                "split_official", "speaker_id", "wav_exists",
                                "raw_translated_english"]
    assert df["clip_id"].tolist() == ["2", "3", "4", "5"]
    assert df["split_official"].tolist()[:3] == ["train", "train", "val"]
    assert pd.isna(df.loc[3, "split_official"])  # missing (pandas NA), not a split name
    assert df["wav_exists"].tolist() == [True, True, True, False]
    first = Path(df.loc[0, "wav_path"])
    assert first.is_absolute() and first.name == "2.wav" and first.parent.name == "training"
    assert pd.isna(df.loc[3, "wav_path"])
    assert df["speaker_id"].isna().all()  # no speaker field in this corpus
    assert df.loc[1, "bn_text"] == "তুমি কোথায়"
    assert df.loc[0, "raw_translated_english"] == "I will go to the market"


def test_ctg_text_is_verbatim_and_nfc_column_is_normalised(root: Path) -> None:
    df = corpus.load_index(root)
    assert df.loc[0, "ctg_text"] == f"আঁই  বাজারত যাই{YA}ুম "  # untouched
    assert df.loc[0, "ctg_text_nfc"] == "আঁই বাজারত যাইয়ুম"
    assert df.loc[2, "ctg_text"] == "\tভালা\n"
    assert df.loc[2, "ctg_text_nfc"] == "ভালা"


def test_test_split_and_default_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    _corpus(paths.data_fast(), {"training": [2], "validation": [3], "test": [4, 5]})
    df = corpus.load_index()
    assert df["split_official"].tolist() == ["train", "val", "test", "test"]


def test_trailing_empty_rows_are_dropped_and_cells_become_text(tmp_path: Path) -> None:
    rows = [*ROWS[:2], ("৩", 3, "তিন"), (None, None, None), (None, None, None)]
    root = _corpus(tmp_path / "c", {"training": [2, 3, 4], "validation": [], "test": []}, rows)
    df = corpus.load_index(root)
    assert df["clip_id"].tolist() == ["2", "3", "4"]
    assert df.loc[2, "raw_translated_english"] == "3"


def test_row_with_wav_but_no_dialect_text_raises(tmp_path: Path) -> None:
    rows = [ROWS[0], ("ক", "k", "   ")]
    root = _corpus(tmp_path / "c", {"training": [2, 3], "validation": [], "test": []}, rows)
    with pytest.raises(ValueError, match=r"no 'Chittagonian Dialect' text.*\[3\]"):
        corpus.load_index(root)


def test_bad_wav_name_and_missing_split_folder_raise(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "c", {"training": [2], "validation": [], "test": []})
    _wav(root / "Dataset" / "Speech" / "training" / "2a.wav")
    with pytest.raises(ValueError, match=r"not an Excel row number: .*2a\.wav"):
        corpus.load_index(root)
    root2 = _corpus(tmp_path / "d", {"training": [2], "validation": [3]})
    (root2 / "Dataset" / "Speech" / "test").rmdir()
    with pytest.raises(FileNotFoundError, match="split folder missing"):
        corpus.load_index(root2)


def test_wav_without_row_raises(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "c", {"training": [2, 3, 9]})
    with pytest.raises(ValueError, match=r"no spreadsheet row.*9"):
        corpus.load_index(root)


def test_same_stem_in_two_splits_raises(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "c", {"training": [2, 3], "test": [3]})
    with pytest.raises(ValueError, match=r"in more than one split.*3"):
        corpus.load_index(root)


def test_unexpected_header_raises(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "c", {"training": [2]}, rows=[])
    xlsx = root / "Dataset" / "Chittagonian_translated.xlsx"
    wb = openpyxl.load_workbook(xlsx)
    wb.active["C1"] = "Dialect"
    wb.save(xlsx)
    with pytest.raises(ValueError, match="Chittagonian Dialect"):
        corpus.load_index(root)


def test_missing_workbook_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Chittagonian_translated.xlsx"):
        corpus.load_index(tmp_path / "nothing")


def test_cli_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    _corpus(paths.data_fast(), {"training": [2, 3], "validation": [4]})
    assert cli.main(["index"]) == 0
    out = capsys.readouterr().out
    assert "train 2" in out and "val 1" in out and "(none) 1" in out
    assert "wav_exists True 3" in out and "wav_exists False 1" in out
    assert "speakers: no speaker field" in out
    assert "PASS: index has 4 rows (3 with audio)" in out


def test_cli_index_fails_without_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    assert cli.main(["index"]) == 1
    assert "FAIL: index:" in capsys.readouterr().out
