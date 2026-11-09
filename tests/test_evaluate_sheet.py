"""Tests for evaluate.make_rating_sheet and `boinga eval-sheet`: blind clips, hidden key, sheet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import cli, evaluate, paths

RATE = 22050
N = 20


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake final/index.csv as C10.2 writes it: A quiet, B loud, lengths unique per wav."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    final = evaluate.final_dir()
    rows = []
    for i in range(1, N + 1):
        pid = f"P{i:02d}"
        for k, (name, level) in enumerate([("A_baseline", 0.05), ("B_boinga", 0.5),
                                           ("R_real", 0.9)]):
            n = RATE + 100 * i + 10 * k  # unique length: identifies the source of a blind clip
            wav = (level * np.sin(np.arange(n) * 0.07)).astype(np.float32)
            path = final / name / f"{pid}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, wav, RATE)
            rows.append({"id": pid, "set": name, "path": str(path), "duration_s": n / RATE,
                         "rms_db": -20.0, "text": f" লেখা {i} "})
    pd.DataFrame(rows).to_csv(final / "index.csv", index=False)
    return final


def _key(final: Path) -> pd.DataFrame:
    return pd.read_csv(final / "key.csv", dtype=str)


def test_sheet_key_and_blind_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    final = _setup(tmp_path, monkeypatch)
    assert evaluate.make_rating_sheet(seed=7) == 40
    blind = final / "blind"
    key = _key(final)
    assert len(key) == 40 and set(key["set"]) == {"A_baseline", "B_boinga"}
    assert not list(blind.glob("*key*")) and "key" not in (blind / "rating_sheet.csv").read_text(
        encoding="utf-8")  # the key never sits next to the clips
    assert (key["clip"] == "clip1").sum() == 20
    assert ((key["clip"] == "clip1") & (key["set"] == "B_boinga")).sum() == 10  # balanced order
    index = pd.read_csv(final / "index.csv")
    length = {Path(p).as_posix(): sf.info(p).frames for p in index["path"]}
    for r in key.itertuples():  # every blind clip is the voice the key names
        clip = blind / f"{r.prompt_id}_{r.clip}.wav"
        src = (final / r.set / f"{r.prompt_id}.wav").as_posix()
        assert sf.info(clip).frames == length[src]
    a, sr = sf.read(blind / "P01_clip1.wav")
    b, _ = sf.read(blind / "P01_clip2.wav")
    rms = [20 * np.log10(np.sqrt(np.mean(x ** 2))) for x in (a, b)]
    assert sr == RATE and abs(rms[0] - rms[1]) < 0.5  # loudness-matched, though sources differ
    assert abs(rms[0] - evaluate.BLIND_RMS_DBFS) < 0.5
    sheet = pd.read_csv(blind / "rating_sheet.csv", dtype=str, keep_default_na=False)
    assert list(sheet.columns) == ["rater", "prompt_id", "clip", "text", "intelligibility",
                                   "naturalness", "dialect_authenticity", "dialect_words",
                                   "comments"]
    assert len(sheet) == 40 and sheet.loc[0, "text"] == " লেখা 1 "  # verbatim text
    assert (sheet[["rater", "intelligibility", "comments"]] == "").all().all()
    assert not list(blind.glob("*R_real*")) and len(list(blind.glob("*.wav"))) == 40
    out = capsys.readouterr().out
    assert "=== BEGIN reports/rater_instructions.md ===" in out
    assert "key sha256" in out and "A_baseline" not in out  # nothing printed reveals the order
    instructions = (blind / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "dialect_authenticity" in instructions and "চাটগাঁইয়া" in instructions
    assert instructions == (paths.REPO / "reports" / "rater_instructions.md").read_text(
        encoding="utf-8")


def test_shuffle_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = _setup(tmp_path, monkeypatch)
    evaluate.make_rating_sheet(seed=7)
    first = _key(final)
    evaluate.make_rating_sheet(seed=7)
    pd.testing.assert_frame_equal(first, _key(final))
    evaluate.make_rating_sheet(seed=8)
    assert not first.equals(_key(final))


def test_filled_sheet_is_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    final = _setup(tmp_path, monkeypatch)
    evaluate.make_rating_sheet(seed=7)
    sheet = final / "blind" / "rating_sheet.csv"
    filled = sheet.read_text(encoding="utf-8").replace(",,,,,", ",4,4,3,3,", 1)
    sheet.write_text(filled, encoding="utf-8")
    evaluate.make_rating_sheet(seed=7)
    assert sheet.read_text(encoding="utf-8") == filled
    assert "WARN: kept the edited" in capsys.readouterr().out
    with pytest.raises(ValueError, match="a different seed"):
        evaluate.make_rating_sheet(seed=8)  # would reshuffle clips under a filled sheet
    sheet.write_text(sheet.read_text(encoding="utf-8").replace(",4,4,3,3,", ",,,,,", 1),
                     encoding="utf-8")  # blind sheet clean again, but a rater file exists
    evaluate.ratings_dir().mkdir(parents=True)
    (evaluate.ratings_dir() / "me.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ratings exist"):
        evaluate.make_rating_sheet(seed=8)


def test_missing_audio_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    final = _setup(tmp_path, monkeypatch)
    assert cli.main(["eval-sheet"]) == 0
    assert "PASS: rating sheet 40 rows" in capsys.readouterr().out
    (final / "B_boinga" / "P05.wav").unlink()
    assert cli.main(["eval-sheet"]) == 1
    out = capsys.readouterr().out
    assert "FAIL: eval-sheet: FileNotFoundError: P05 clip" in out and "B_boinga" not in out
    sf.write(final / "B_boinga" / "P05.wav", np.zeros(RATE, dtype=np.float32), RATE)
    assert cli.main(["eval-sheet"]) == 1
    out = capsys.readouterr().out
    assert "empty or silent audio" in out and "B_boinga" not in out
