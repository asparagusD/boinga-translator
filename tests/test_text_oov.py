"""Tests for text_oov.py, speaker_select.load_selection and `boinga oov` on a fake charset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from boinga_translator import cli, corpus, paths, speaker_select, text_frontend, text_oov

REAL_MAP = Path(__file__).resolve().parents[1] / "configs" / text_frontend.CHAR_MAP_FILE

FAKE = frozenset("abcdন ")  # a tiny "model charset"; the space is never reported


def test_oov_report_counts_clips_and_examples() -> None:
    texts = ["abc xyz", "ax ax ax", "ab", "x?", "অন y"]
    df = text_oov.oov_report(texts, FAKE)
    assert list(df.columns) == list(text_oov.OOV_COLS)
    assert list(df["char"]) == ["x", "y", "?", "z", "অ"]  # count desc, then code point
    x = df.set_index("char").loc["x"]
    assert (x["count"], x["clips_affected"]) == (5, 3)
    assert x["codepoint"] == "U+0078" and x["name"] == "LATIN SMALL LETTER X"
    assert x["examples"] == ["xyz", "ax", "x?"]  # 3 distinct words, first seen first
    assert df.set_index("char").loc["অ", "name"] == "BENGALI LETTER A"


def test_oov_report_empty_when_everything_is_known() -> None:
    df = text_oov.oov_report(["abc", "  dন "], FAKE)
    assert df.empty and list(df.columns) == list(text_oov.OOV_COLS)


def test_render_rates_and_escaping() -> None:
    texts = ["ab|", "cd", "abcd"]  # 1 unknown char of 9 non-space chars, 1 of 3 clips
    md = text_oov.render(texts, FAKE, voice="male", n_excluded=2)
    assert "| \\| | U+007C | VERTICAL LINE | 1 | 1 | ab\\| |" in md
    assert "1 of 9 characters (11.11%)" in md
    assert "1 of 3 clips (33.33%)" in md
    assert "base voice **male**" in md and "2 selected clips excluded" in md
    clean = text_oov.render(["ab"], FAKE, voice="male", n_excluded=0)
    assert "No character is missing" in clean and "0 of 2 characters (0.00%)" in clean
    tricky = text_oov.render(["a`b *x_"], FAKE, voice="male", n_excluded=0)
    assert "| \\` | U+0060 | GRAVE ACCENT | 1 | 1 | a\\`b |" in tricky
    assert "| \\* | U+002A | ASTERISK | 1 | 1 | \\*x\\_ |" in tricky
    empty = text_oov.render([], FAKE, voice="male", n_excluded=4)
    assert "0 of 0 characters (0.00%)" in empty and "0 of 0 clips (0.00%)" in empty


def test_unicode_whitespace_is_never_reported() -> None:
    texts = ["ab cd\tab d\n"]
    assert text_oov.oov_report(texts, FAKE).empty
    assert text_oov.clips_with_oov(texts, FAKE) == 0


@pytest.fixture
def fake_drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pd.DataFrame]:
    """Fake repo + Drive: a 4-clip selection (base voice male) and a matching index."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    sel = pd.DataFrame({"clip_id": ["0386", "2", "3", "4"], "split_official": "train",
                        "duration_s": 3.0})
    data = sel.to_csv(index=False, lineterminator="\n").encode("utf-8")
    paths.speaker_selection().parent.mkdir(parents=True)
    paths.speaker_selection().write_bytes(data)
    cfg = paths.REPO / speaker_select.SELECTION_REL
    cfg.parent.mkdir(parents=True)
    (cfg.parent / text_frontend.CHAR_MAP_FILE).write_bytes(REAL_MAP.read_bytes())
    text_frontend.default_char_map.cache_clear()  # read the map from the fake repo
    (paths.REPO / "configs" / "pretrained_facts.json").write_text(json.dumps(
        {"male": {"characters": "আইযাম", "punctuations": "."}}), encoding="utf-8")
    cfg.write_text(json.dumps({"kind": "cluster", "id": 1, "base_voice": "male",
                               "csv_sha256": hashlib.sha256(data).hexdigest()}),
                   encoding="utf-8")
    index = pd.DataFrame({"clip_id": ["0386", "2", "3", "4", "99"],
                          "ctg_text_nfc": ["আই যাম।", "আইক", "২০ আই", "মা", "ক্ষ"]})
    monkeypatch.setattr(corpus, "load_index", lambda: index)
    yield index
    text_frontend.default_char_map.cache_clear()  # don't leak the fake repo's map


def test_load_selection_checks_the_csv_hash(fake_drive: pd.DataFrame) -> None:
    rows, sel = speaker_select.load_selection()
    assert list(rows["clip_id"]) == ["0386", "2", "3", "4"]  # ids stay text
    assert sel["base_voice"] == "male"
    paths.speaker_selection().write_bytes(b"clip_id,split_official,duration_s\n2,train,3.0\n")
    with pytest.raises(ValueError, match="sha256"):
        speaker_select.load_selection()


@pytest.mark.parametrize("break_it", ["no_csv", "no_sha"])
def test_cli_oov_fails_cleanly_on_a_broken_selection(
        fake_drive: pd.DataFrame, break_it: str, capsys: pytest.CaptureFixture[str]) -> None:
    if break_it == "no_csv":
        paths.speaker_selection().unlink()
    else:
        cfg = paths.REPO / speaker_select.SELECTION_REL
        cfg.write_text(json.dumps({"base_voice": "male"}), encoding="utf-8")
    assert cli.main(["oov"]) == 1
    expected = "FileNotFoundError" if break_it == "no_csv" else "KeyError"
    assert f"FAIL: oov: {expected}" in capsys.readouterr().out


def test_cli_oov_writes_and_emits_the_report(fake_drive: pd.DataFrame,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["oov"]) == 0
    out = capsys.readouterr().out
    assert "=== BEGIN reports/oov_report.md ===" in out
    assert "PASS: oov report: 1 missing character(s) in 1 of 3 clips" in out
    md = (paths.REPO / "reports" / "oov_report.md").read_text(encoding="utf-8")
    assert "| ক | U+0995 | BENGALI LETTER KA | 1 | 1 | আইক |" in md  # clip 99 not selected
    assert "1 selected clips excluded" in md  # "২০ আই": digits exclude the row (char_map)
    assert "base voice **male**" in md
    assert (paths.drive_reports() / "reports" / "oov_report.md").is_file()


def test_cli_oov_fails_on_a_clip_missing_from_the_index(
        fake_drive: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(corpus, "load_index", lambda: fake_drive[fake_drive["clip_id"] != "2"])
    assert cli.main(["oov"]) == 1
    assert "FAIL: oov: ValueError: 1 selected clip(s) not in the corpus index" in (
        capsys.readouterr().out)
