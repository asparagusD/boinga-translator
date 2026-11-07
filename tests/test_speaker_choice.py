"""Tests for speaker_select listening and the saved choice: listen_clips, save_selection, form."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from boinga_translator import cli, data_audit, paths, speaker_select
from test_audit_summary import _rows  # clean clips c0..c7, 3.0 s each; c5 val, c6/c7 test

SEL_REL = "configs/speaker_selection.json"


@pytest.fixture
def audit_df(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Fake audit table on the fake Drive plus a cluster file: 1 = c0..c4, 2 = c5..c7."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    out = paths.drive_reports() / data_audit.AUDIT_CSV
    out.parent.mkdir(parents=True)
    pd.DataFrame(_rows(), columns=list(data_audit.AUDIT_COLS)).to_csv(out, index=False)
    df = data_audit.read_audit(out)
    labels = df[df["clip_id"].isin([f"c{i}" for i in range(8)])][
        ["clip_id", "split_official", "duration_s"]]
    labels.assign(cluster=[1, 1, 1, 1, 1, 2, 2, 2], threshold=0.3).to_csv(
        paths.speaker_clusters(), index=False)
    return df


def test_listen_clips_are_seeded_labelled_and_from_the_candidate(audit_df: pd.DataFrame) -> None:
    wavs, labels = speaker_select.listen_clips(1, n=3, audit_df=audit_df)
    assert len(wavs) == len(labels) == 3
    ids = [Path(w).stem for w in wavs]
    assert set(ids) <= {"c0", "c1", "c2", "c3", "c4"}
    assert speaker_select.listen_clips(1, n=3, audit_df=audit_df)[0] == wavs  # same seed
    assert labels[0].startswith(f"{ids[0]} [train] CTG: লেখা {ids[0]}")  # verbatim text
    all_wavs, _ = speaker_select.listen_clips(2, n=10, audit_df=audit_df)
    assert sorted(Path(w).stem for w in all_wavs) == ["c5", "c6", "c7"]  # n > size: all
    with pytest.raises(ValueError, match="no cluster 9"):
        speaker_select.listen_clips(9, audit_df=audit_df)


def test_save_selection_writes_csv_and_json(audit_df: pd.DataFrame,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    sel = speaker_select.save_selection(2, base_voice="male")
    csv = paths.speaker_selection()
    saved = pd.read_csv(csv, dtype={"clip_id": str})
    assert list(saved["clip_id"]) == ["c5", "c6", "c7"]
    assert list(saved.columns) == ["clip_id", "split_official", "duration_s"]
    on_disk = json.loads((paths.REPO / SEL_REL).read_text(encoding="utf-8"))
    assert on_disk == sel
    assert sel == {"kind": "cluster", "id": 2, "n_clips": 3, "clean_minutes": 0.15,
                   "minutes_by_official_split": {"train": 0.0, "val": 0.05, "test": 0.1},
                   "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                   "base_voice": "male"}
    out = capsys.readouterr().out
    assert f"=== BEGIN {SEL_REL} ===" in out and "PASS: selection saved" in out


def test_numeric_looking_clip_ids_keep_their_text(audit_df: pd.DataFrame) -> None:
    pd.DataFrame({"clip_id": ["0386", "29"], "split_official": ["test", "train"],
                  "duration_s": [3.0, 3.0], "cluster": [1, 1], "threshold": 0.3}).to_csv(
        paths.speaker_clusters(), index=False)
    speaker_select.save_selection(1, base_voice="female")
    lines = paths.speaker_selection().read_text(encoding="utf-8").splitlines()
    assert lines[1:] == ["0386,test,3.0", "29,train,3.0"]  # "0386" is not turned into 386


def test_clips_outside_the_official_splits_are_refused(audit_df: pd.DataFrame) -> None:
    pd.DataFrame({"clip_id": ["1"], "split_official": ["extra"], "duration_s": [3.0],
                  "cluster": [1], "threshold": 0.3}).to_csv(paths.speaker_clusters(), index=False)
    with pytest.raises(ValueError, match="outside"):
        speaker_select.save_selection(1, base_voice="female")


@pytest.mark.parametrize(("cid", "voice", "match"), [
    (9, "female", "no cluster 9"), (1, "child", "base_voice")])
def test_save_selection_refuses_bad_input(audit_df: pd.DataFrame, cid: int, voice: str,
                                          match: str) -> None:
    with pytest.raises(ValueError, match=match):
        speaker_select.save_selection(cid, base_voice=voice)
    assert not paths.speaker_selection().exists()


@pytest.mark.parametrize(("chosen", "one_voice", "gender"), [
    ("", "yes", "female"), ("1", "", "female"), ("1", "yes", ""), ("1", "no", "female")])
def test_form_without_a_complete_yes_saves_nothing(
        audit_df: pd.DataFrame, capsys: pytest.CaptureFixture[str],
        chosen: str, one_voice: str, gender: str) -> None:
    speaker_select.save_from_form(chosen, one_voice, gender)
    out = capsys.readouterr().out
    assert "NOTES-MISSING choice incomplete" in out and "PASS" not in out
    assert not paths.speaker_selection().exists()


def test_form_saves_a_complete_choice(audit_df: pd.DataFrame,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    speaker_select.save_from_form(" 1 ", "yes", "female")
    assert "PASS: selection saved" in capsys.readouterr().out
    sel = json.loads((paths.REPO / SEL_REL).read_text(encoding="utf-8"))
    assert sel["id"] == 1 and sel["base_voice"] == "female" and sel["n_clips"] == 5
    speaker_select.save_from_form("one", "yes", "female")
    assert "FAIL: CHOSEN_ID must be a cluster number" in capsys.readouterr().out


def test_play_candidates_prints_a_heading_per_candidate(
        audit_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    speaker_select.play_candidates(3, clips=2)
    out = capsys.readouterr().out
    assert "=== candidate 1: 0.25 clean min, 5 clips ===" in out
    assert "=== candidate 2: 0.15 clean min, 3 clips ===" in out
    assert out.count("AUDIO ") == 4  # outside IPython play() prints one line per clip


def test_cli_save_speaker(audit_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["save-speaker", "--id", "1", "--base-voice", "female"]) == 0
    assert "PASS: selection saved" in capsys.readouterr().out
    assert cli.main(["save-speaker", "--id", "9", "--base-voice", "female"]) == 1
    assert "FAIL: save-speaker: ValueError" in capsys.readouterr().out
