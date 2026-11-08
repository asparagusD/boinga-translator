"""Tests for rating_app: start/resume a rater's sheet, record scores, progress, check (no widgets)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from boinga_translator import evaluate, paths, rating_app

IDS = ["P01", "P02"]


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The blind folder as C10.3 leaves it: empty rating_sheet.csv for 2 prompts."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    blind = evaluate.blind_dir()
    blind.mkdir(parents=True)
    rows = [{"prompt_id": p, "clip": c, "text": f" লেখা {p} "} for p in IDS
            for c in ("clip1", "clip2")]
    pd.DataFrame(rows).reindex(columns=evaluate.SHEET_COLS, fill_value="").to_csv(
        blind / "rating_sheet.csv", index=False)
    return blind


def _rate(sheet: pd.DataFrame, pid: str, value: int) -> pd.DataFrame:
    scores = {c: value for c in evaluate.CRITERIA}
    sheet = rating_app.record(sheet, pid, "clip1", scores, "  too\nfast ")
    return rating_app.record(sheet, pid, "clip2", scores, "")


def test_start_record_save_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    sheet = rating_app.load_or_start("Me ")
    assert len(sheet) == 4 and set(sheet["rater"]) == {"me"}
    assert rating_app.progress(sheet) == (0, 4) and rating_app.next_prompt(sheet) == "P01"
    sheet = _rate(sheet, "P01", 4)
    rating_app.save(sheet, "me")
    path = evaluate.ratings_dir() / "me.csv"
    assert path.is_file() and not list(path.parent.glob("*.tmp"))
    again = rating_app.load_or_start("me")  # resume from Drive, not a fresh copy
    assert rating_app.progress(again) == (2, 4) and rating_app.next_prompt(again) == "P02"
    row = again[(again["prompt_id"] == "P01") & (again["clip"] == "clip1")].iloc[0]
    assert row["naturalness"] == "4" and row["comments"] == "too fast"
    assert row["text"] == " লেখা P01 "  # verbatim
    done = _rate(again, "P02", 2)
    assert rating_app.progress(done) == (4, 4) and rating_app.next_prompt(done) is None


def test_record_refuses_bad_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    sheet = rating_app.load_or_start("me")
    good = {c: 3 for c in evaluate.CRITERIA}
    with pytest.raises(ValueError, match="naturalness must be 1-5"):
        rating_app.record(sheet, "P01", "clip1", {**good, "naturalness": 6}, "")
    with pytest.raises(ValueError, match="dialect_words must be 1-5"):
        rating_app.record(sheet, "P01", "clip1", {**good, "dialect_words": None}, "")
    with pytest.raises(KeyError, match="P09 clip1"):
        rating_app.record(sheet, "P09", "clip1", good, "")
    with pytest.raises(ValueError, match="rater name"):
        rating_app.load_or_start(" / ")


def test_check_needs_every_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        rating_app.check("me")
    assert "FAIL: no ratings saved for 'me'" in capsys.readouterr().out
    sheet = _rate(rating_app.load_or_start("me"), "P01", 5)
    rating_app.save(sheet, "me")
    with pytest.raises(SystemExit):
        rating_app.check("me")
    assert "FAIL: rated 2 of 4 clips (next: P02)" in capsys.readouterr().out
    rating_app.save(_rate(sheet, "P02", 1), "me")
    rating_app.check("me")
    assert "PASS: rated 4 of 4 clips" in capsys.readouterr().out


def test_rate_builds_the_widgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ipywidgets")  # Colab has it; the local dev env may not
    blind = _setup(tmp_path, monkeypatch)
    for p in IDS:
        for c in ("clip1", "clip2"):
            (blind / f"{p}_{c}.wav").write_bytes(b"RIFF")
    head, columns, buttons, msg = rating_app.rate("me").children
    back, ahead = buttons.children
    assert "P01 (1 of 2)" in head.value and " লেখা P01 " in head.value
    ahead.click()  # nothing picked yet: refused, nothing saved
    assert "must be 1-5" in msg.value and not rating_app.sheet_path("me").exists()
    for col in columns.children:
        *_, note = col.children
        for picker in col.children[2:-1]:
            picker.value = "4"
        note.value = "ok"
    ahead.click()
    assert msg.value == "saved P01" and "P02 (2 of 2)" in head.value
    saved = rating_app.load_or_start("me")
    assert rating_app.progress(saved) == (2, 4)
    back.click()
    assert "P01 (1 of 2)" in head.value and columns.children[0].children[2].value == "4"
