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


def test_check_adopts_ratings_typed_into_the_blind_sheet(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    blind = _setup(tmp_path, monkeypatch)
    sheet = pd.read_csv(blind / "rating_sheet.csv", dtype=str, keep_default_na=False)
    rating_app.save(_rate(rating_app.load_or_start("me"), "P01", 2), "me")  # partial, saved
    filled = _rate(_rate(sheet, "P01", 4), "P02", 5)
    filled.to_csv(blind / "rating_sheet.csv", index=False)  # scores typed into the blind sheet
    rating_app.check("me")
    out = capsys.readouterr().out
    assert "copied 4 rated clips from blind/rating_sheet.csv" in out
    assert "previous sheet kept as me.csv." in out and "PASS: rated 4 of 4 clips" in out
    saved = rating_app.load_or_start("me")
    assert set(saved["rater"]) == {"me"} and set(saved["naturalness"]) == {"4", "5"}
    assert len(list(evaluate.ratings_dir().glob("me.csv.*.bak"))) == 1
    assert len(list(evaluate.ratings_dir().glob("*.csv"))) == 1  # the backup is never scored
    rating_app.check("me")  # second run: nothing more to copy
    assert "copied" not in capsys.readouterr().out


class _Resp:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> object:
        return self.payload

    @property
    def content(self) -> bytes:
        return str(self.payload).encode("utf-8")


class _FakeDrive:
    """Drive API stand-in: a file listing, then the CSV export of the sheet."""

    def __init__(self, files: list[dict[str, str]], csv: str = "") -> None:
        self.files, self.csv, self.calls = files, csv, []

    def get(self, url: str, params: dict[str, str] | None = None) -> _Resp:
        self.calls.append((url, params or {}))
        if url.endswith("/export"):
            return _Resp(self.csv)
        return _Resp({"files": self.files})


def test_import_google_sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:
    blind = _setup(tmp_path, monkeypatch)
    sheet = pd.read_csv(blind / "rating_sheet.csv", dtype=str, keep_default_na=False)
    csv = _rate(_rate(sheet, "P01", 3), "P02", 1).assign(rater="Some One").to_csv(index=False)
    (blind / "rating_sheet.csv").unlink()  # the CSV became a Google Sheet on Drive
    rating_app.import_google_sheet("me", session=_FakeDrive([], ""))
    assert "no Google Sheet called 'rating_sheet'" in capsys.readouterr().out
    two = _FakeDrive([{"id": "a"}, {"id": "b"}])
    with pytest.raises(ValueError, match="2 Google Sheets called 'rating_sheet'"):
        rating_app.import_google_sheet("me", session=two)
    drive = _FakeDrive([{"id": "abc"}], csv)
    rating_app.import_google_sheet("me", session=drive)
    assert drive.calls[-1][0].endswith("/files/abc/export")
    assert drive.calls[-1][1] == {"mimeType": "text/csv"}
    assert "copied 4 rated clips from Google Sheet 'rating_sheet'" in capsys.readouterr().out
    saved = rating_app.load_or_start("me")
    assert set(saved["rater"]) == {"me"} and rating_app.progress(saved) == (4, 4)
    rating_app.check("me")  # no blind CSV any more: check still passes on the imported sheet
    assert "PASS: rated 4 of 4 clips" in capsys.readouterr().out
    bad = _FakeDrive([{"id": "abc"}], "prompt_id,clip,naturalness\nP01,clip1,4\n")
    with pytest.raises(ValueError, match="missing columns"):
        rating_app.import_google_sheet("me", session=bad)


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
