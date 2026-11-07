"""Tests for data_audit.audit() and `boinga audit`: per-row stats, duplicates, resume, CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from boinga_translator import cli, corpus, data_audit, paths
from test_corpus import ROWS, _corpus  # tiny fake corpus in the real layout

VOICE = "female"


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    rows = [*ROWS, ROWS[1]]  # Excel row 6 repeats row 3's text
    _corpus(paths.data_fast(), {"training": [2, 3, 6], "validation": [4]}, rows)
    return corpus.load_index()  # row 5 has no wav


def test_audit_writes_one_row_per_index_row(index: pd.DataFrame, tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "audit.csv"
    df = data_audit.audit(index, VOICE, out, processes=1)
    assert sorted(df["clip_id"]) == ["2", "3", "4", "5", "6"]
    assert set(data_audit.AUDIO_KEYS) | set(data_audit.TEXT_KEYS) <= set(df.columns)
    assert {"chars_per_second", "duplicate_text", "split_official", "ctg_text_nfc"} <= set(
        df.columns)
    by = df.set_index("clip_id")
    assert bool(by.loc["5", "decodable"]) is False and by.loc["5", "error"] == "no wav file"
    assert bool(by.loc["2", "decodable"]) is True
    assert by.loc["3", "duplicate_text"] and by.loc["6", "duplicate_text"]
    assert not by.loc["2", "duplicate_text"]
    expected = data_audit.text_stats(by.loc["2", "ctg_text_nfc"],
                                     data_audit.load_model_charset(VOICE))
    assert by.loc["2", "chars_not_in_model"] == "".join(expected["chars_not_in_model"])
    assert by.loc["2", "chars_per_second"] > 0
    out_text = capsys.readouterr().out
    assert "wrote 5 rows" in out_text and "PASS: audited 5 clips" in out_text
    assert pd.read_csv(out, dtype={"clip_id": str})["clip_id"].tolist() == df["clip_id"].tolist()


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = data_audit._audit_one
    monkeypatch.setattr(data_audit, "_audit_one", lambda job: seen.append(job[0]) or real(job))
    return seen


def test_resume_ignores_what_a_dropped_session_left_behind(
        index: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "audit.csv"
    fresh = data_audit.audit(index, VOICE, tmp_path / "fresh.csv", processes=1)
    data_audit.audit(index.iloc[:2], VOICE, out, processes=1)  # rows 2, 3 in a whole part
    parts = tmp_path / "audit.parts"
    (parts / "part-00001.csv.tmp").write_text('clip_id,wav_path\n9,"x, y', encoding="utf-8")
    (parts / "part-00002.csv.tmp").write_bytes(b"")  # cut inside a quote / 0 bytes
    out.write_text("clip_id,split_official\n2,tra", encoding="utf-8")  # cut in a last column
    seen = _spy(monkeypatch)
    capsys.readouterr()
    df = data_audit.audit(index, VOICE, out, processes=1)
    assert sorted(seen) == ["4", "5", "6"]
    assert "skipped 2 rows already in" in capsys.readouterr().out
    pd.testing.assert_frame_equal(df, fresh)  # resumed == fresh, dtypes included
    text = out.read_text(encoding="utf-8")
    assert text.count("clip_id,") == 1  # one header, file rewritten whole
    pd.testing.assert_frame_equal(pd.read_csv(out, dtype={"clip_id": str})["clip_id"].to_frame(),
                                  fresh["clip_id"].to_frame())


def test_rows_whose_text_or_wav_changed_are_redone(
        index: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "audit.csv"
    data_audit.audit(index, VOICE, out, processes=1)
    changed = index.copy()
    changed.loc[changed["clip_id"] == "2", "ctg_text_nfc"] = "নতুন লেখা"
    seen = _spy(monkeypatch)
    df = data_audit.audit(changed, VOICE, out, processes=1)
    assert seen == ["2"]
    assert df.set_index("clip_id").loc["2", "ctg_text_nfc"] == "নতুন লেখা"
    assert df["clip_id"].is_unique and len(df) == len(changed)


def test_other_voice_redoes_rows_with_its_own_charset(
        index: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Female lacks ZWJ (U+200D), male has it: female stats must not be reused for male."""
    zwj = index.copy()
    zwj.loc[zwj["clip_id"] == "2", "ctg_text_nfc"] = "র‍্য"
    out = tmp_path / "audit.csv"
    female = data_audit.audit(zwj, "female", out, processes=1).set_index("clip_id")
    assert "‍" in female.loc["2", "chars_not_in_model"]
    seen = _spy(monkeypatch)
    male = data_audit.audit(zwj, "male", out, processes=1).set_index("clip_id")
    assert sorted(seen) == ["2", "3", "4", "5", "6"]
    assert "‍" not in male.loc["2", "chars_not_in_model"]
    assert male["charset_id"].str.startswith("male-").all()
    seen.clear()
    data_audit.audit(zwj, "male", out, processes=1)
    assert seen == []  # same voice again: nothing redone


def test_audit_fails_when_a_result_is_missing(
        index: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_audit, "_write_part", lambda parts, rows: None)  # nothing saved
    with pytest.raises(RuntimeError, match=r"audit incomplete: 5 index rows"):
        data_audit.audit(index, VOICE, tmp_path / "audit.csv", processes=1)


def test_pool_of_two_matches_inline(index: pd.DataFrame, tmp_path: Path) -> None:
    a = data_audit.audit(index, VOICE, tmp_path / "a.csv", processes=1)
    b = data_audit.audit(index, VOICE, tmp_path / "b.csv", processes=2)
    assert list(a.columns) == list(data_audit.AUDIT_COLS)
    pd.testing.assert_frame_equal(a, b)


def test_chars_per_second() -> None:
    assert data_audit.chars_per_second("আমি বাজারে", 2.0, 0.5, 0.5) == pytest.approx(9.0)
    assert data_audit.chars_per_second("আমি", 1.0, 0.6, 0.6) is None  # no speech left
    assert data_audit.chars_per_second("আমি", None, None, None) is None


def test_cli_audit_default_voice_and_drive_csv(index: pd.DataFrame,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["audit"]) == 0
    out = capsys.readouterr().out
    assert "voice female" in out and "PASS: audited 5 clips" in out
    assert (paths.drive_reports() / "audit.csv").is_file()
    assert cli.main(["audit", "--voice", "male"]) == 0
    assert "voice male" in capsys.readouterr().out


def test_cli_audit_fails_without_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    assert cli.main(["audit"]) == 1
    assert "FAIL: audit:" in capsys.readouterr().out
