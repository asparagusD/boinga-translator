"""Tests for data_audit.audit() and `boinga audit`: per-row stats, duplicates, resume, CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from boinga_translator import audit_summary, cli, corpus, data_audit, paths, text_frontend
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


def test_other_voice_or_char_map_redoes_rows(
        index: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stats depend on the voice's charset and on the char map: either change -> redo."""
    out = tmp_path / "audit.csv"
    female = data_audit.audit(index, "female", out, processes=1)
    seen = _spy(monkeypatch)
    male = data_audit.audit(index, "male", out, processes=1)
    assert sorted(seen) == ["2", "3", "4", "5", "6"]
    assert male["charset_id"].str.startswith("male-").all()
    assert set(male["charset_id"]) != set(female["charset_id"])
    seen.clear()
    data_audit.audit(index, "male", out, processes=1)
    assert seen == []  # same voice and map again: nothing redone
    other = text_frontend.CharMap((("ক", "খ"),), frozenset("#"))
    monkeypatch.setattr(text_frontend, "default_char_map", lambda: other)
    data_audit.audit(index, "male", out, processes=1)
    assert sorted(seen) == ["2", "3", "4", "5", "6"]  # new map: every row redone


def test_parts_from_an_older_audit_are_redone_not_crashed_on(
        index: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run-2 parts on Drive lack ctg_text_model/excluded_char and use the old charset_id."""
    parts = tmp_path / "audit.parts"
    parts.mkdir()
    old_cols = [c for c in data_audit._PART_COLS if c not in ("ctg_text_model", "excluded_char")]
    old = pd.DataFrame([{**dict.fromkeys(old_cols, ""), "clip_id": cid, "charset_id": "female-abc",
                         "ctg_text_nfc": text, "wav_path": wav, "decodable": "True"}
                        for cid, text, wav in zip(index["clip_id"], index["ctg_text_nfc"],
                                                  index["wav_path"].fillna(""), strict=True)])
    old.to_csv(parts / "part-00000.csv", index=False)
    seen = _spy(monkeypatch)
    df = data_audit.audit(index, VOICE, tmp_path / "audit.csv", processes=1)
    assert sorted(seen) == ["2", "3", "4", "5", "6"]  # all stale: redone, no KeyError
    assert (df["ctg_text_model"] != "").sum() == 5


def test_summary_of_a_pre_char_map_audit_says_rerun(index: pd.DataFrame, tmp_path: Path) -> None:
    df = data_audit.audit(index, VOICE, tmp_path / "audit.csv", processes=1)
    with pytest.raises(ValueError, match="re-run `boinga audit`"):
        audit_summary.exclusion_reasons(df.drop(columns=["excluded_char"]))


def test_stats_are_on_the_mapped_text(index: pd.DataFrame, tmp_path: Path) -> None:
    rows = index.copy()
    rows.loc[rows["clip_id"] == "2", "ctg_text_nfc"] = "দু’রাত (আর) গেলাম।"
    rows.loc[rows["clip_id"] == "3", "ctg_text_nfc"] = "২০১৯ সালর কথা।"
    df = data_audit.audit(rows, "female", tmp_path / "audit.csv", processes=1)
    by = df.set_index("clip_id")
    assert by.loc["2", "ctg_text_model"] == "দু’রাত আর গেলাম."
    assert by.loc["2", "chars_not_in_model"] == "" and not by.loc["2", "excluded_char"]
    assert by.loc["3", "excluded_char"] and by.loc["3", "has_digits"]


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
