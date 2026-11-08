"""Tests for evaluate.score and `boinga eval-score`: validation, unblinding and reports/eval.md."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from boinga_translator import cli, evaluate, paths

CRITERIA = ["intelligibility", "naturalness", "dialect_authenticity", "dialect_words"]
IDS = ["P01", "P02", "P03"]


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Key: P01/P03 clip1 = baseline, P02 clip1 = boinga. Raters prefer boinga except on P03."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    final = evaluate.final_dir()
    final.mkdir(parents=True)
    key = []
    for pid in IDS:
        first = "B_boinga" if pid == "P02" else "A_baseline"
        second = "A_baseline" if first == "B_boinga" else "B_boinga"
        key += [{"prompt_id": pid, "clip": "clip1", "set": first, "source": "x"},
                {"prompt_id": pid, "clip": "clip2", "set": second, "source": "y"}]
    pd.DataFrame(key).to_csv(final / "key.csv", index=False)
    ratings = final / "ratings"
    ratings.mkdir()
    for name, bump in (("alice", 0), ("bob", 1)):
        rows = []
        for k in key:
            boinga = k["set"] == "B_boinga"
            score = (4 if boinga else 2) if k["prompt_id"] != "P03" else (3 if boinga else 3 + bump)
            rows.append({"rater": name, "prompt_id": k["prompt_id"], "clip": k["clip"],
                         "text": "লেখা", **{c: str(score) for c in CRITERIA},
                         "comments": "robotic" if not boinga and k["prompt_id"] == "P01" else ""})
        pd.DataFrame(rows, columns=evaluate.SHEET_COLS).to_csv(ratings / f"{name}.csv",
                                                               index=False)
    return ratings


def test_score_unblinds_and_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    _setup(tmp_path, monkeypatch)
    assert evaluate.score() == 2
    report = (paths.REPO / "reports" / "eval.md").read_text(encoding="utf-8")
    assert "=== BEGIN reports/eval.md ===" in capsys.readouterr().out
    assert "2 raters (R1, R2), 3 prompts, 12 ratings" in report
    assert "alice" not in report and "bob" not in report  # raters are anonymised
    # intelligibility: A = (2,2,3 | 2,2,4) -> 2.50 +- 0.84; B = (4,4,3 | 4,4,3) -> 3.67 +- 0.52
    assert "| intelligibility | 2.50 ± 0.84 | 3.67 ± 0.52 | +1.17 |" in report
    assert "| P01 | 2.00 | 4.00 | 1.00 |" in report  # B wins for both raters
    assert "| P03 | 3.50 | 3.00 | 0.25 |" in report  # tie for alice (0.5), loss for bob (0)
    assert "B_boinga win rate over prompts: 0.75" in report  # (1 + 1 + 0.25) / 3
    assert "- P01 (R1): robotic" in report.split("### A_baseline")[1].split("###")[0]
    assert "## Limitations" in report and "WER" in report and "Indian Bengali" in report


def test_bad_cells_are_listed_never_guessed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    ratings = _setup(tmp_path, monkeypatch)
    df = pd.read_csv(ratings / "bob.csv", dtype=str, keep_default_na=False)
    df.loc[0, "naturalness"] = "4.5"
    df.loc[3, "dialect_words"] = ""
    df.loc[5, "intelligibility"] = "6"
    df.loc[1, "naturalness"] = "4.0"  # not a whole number as written: rejected, never rounded
    df.loc[2, "naturalness"] = " 4 "  # padding only: accepted
    df.to_csv(ratings / "bob.csv", index=False)
    with pytest.raises(ValueError) as err:
        evaluate.score()
    msg = str(err.value)
    assert "4 bad cell(s)" in msg and "row 3 naturalness: '4.0'" in msg
    assert "row 4 naturalness" not in msg
    assert "bob.csv row 2 naturalness: '4.5'" in msg  # row = spreadsheet row (header is 1)
    assert "bob.csv row 5 dialect_words: ''" in msg
    assert "bob.csv row 7 intelligibility: '6'" in msg
    assert not (paths.REPO / "reports" / "eval.md").exists()
    assert cli.main(["eval-score"]) == 1
    assert "FAIL: eval-score:" in capsys.readouterr().out


def test_sheet_must_cover_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ratings = _setup(tmp_path, monkeypatch)
    df = pd.read_csv(ratings / "alice.csv", dtype=str, keep_default_na=False)
    df.iloc[:-1].to_csv(ratings / "alice.csv", index=False)  # P03 clip2 row deleted
    with pytest.raises(ValueError, match=r"alice.csv: missing rows \['P03 clip2'\]"):
        evaluate.score()
    df.assign(clip=df["clip"].replace("clip2", "clip9")).to_csv(ratings / "alice.csv",
                                                                index=False)
    with pytest.raises(ValueError, match="not in the key"):
        evaluate.score()
    dup = pd.concat([df, df.iloc[[0]].assign(naturalness="1")])  # a repeated row
    dup.to_csv(ratings / "alice.csv", index=False)
    with pytest.raises(ValueError, match=r"alice.csv: duplicate rows \['P01 clip1'\]"):
        evaluate.score()
    df.drop(columns="comments").to_csv(ratings / "alice.csv", index=False)
    with pytest.raises(ValueError, match=r"missing columns \['comments'\]"):
        evaluate.score()


def test_no_ratings_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                            capsys: pytest.CaptureFixture[str]) -> None:
    ratings = _setup(tmp_path, monkeypatch)
    assert cli.main(["eval-score"]) == 0
    assert "PASS: scored 2 raters" in capsys.readouterr().out
    for f in ratings.glob("*.csv"):
        f.unlink()
    with pytest.raises(FileNotFoundError, match="no rating sheets"):
        evaluate.score()
