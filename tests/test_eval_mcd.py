"""Tests for eval_mcd: MCD with DTW, the per-prompt table in eval.md, `boinga eval-mcd`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import cli, eval_mcd, evaluate, paths

RATE = 22050


def _tone(freqs: tuple[float, ...], seconds: float = 1.0, lead: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    wav = sum(np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) for f in freqs)
    wav = (0.3 * wav / len(freqs)).astype(np.float32)
    return np.concatenate([np.zeros(int(lead * RATE), np.float32), wav])


def _write(path: Path, wav: np.ndarray, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, rate)
    return path


def test_identical_is_zero_and_shift_is_small(tmp_path: Path) -> None:
    ref = _write(tmp_path / "r.wav", _tone((220, 660)))
    same = _write(tmp_path / "a.wav", _tone((220, 660)))
    shifted = _write(tmp_path / "s.wav", _tone((220, 660), lead=0.15))  # starts 150 ms late
    other = _write(tmp_path / "o.wav", _tone((1500, 3100)))
    assert eval_mcd.mcd_dtw(same, ref) == pytest.approx(0.0, abs=1e-6)
    near, far = eval_mcd.mcd_dtw(shifted, ref), eval_mcd.mcd_dtw(other, ref)
    assert 0 < near < 0.5 * far  # DTW absorbs the delay; different audio stays far
    assert eval_mcd.mcd_dtw(other, ref) == pytest.approx(eval_mcd.mcd_dtw(ref, other), rel=0.2)


def test_rate_mismatch_is_resampled(tmp_path: Path) -> None:
    ref = _write(tmp_path / "r.wav", _tone((220, 660)))
    t = np.arange(16000) / 16000
    low = (0.3 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 660 * t)) / 2
           * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    other = _write(tmp_path / "o.wav", _tone((1500, 3100)))
    resampled = eval_mcd.mcd_dtw(_write(tmp_path / "l.wav", low, 16000), ref)
    assert resampled < 0.5 * eval_mcd.mcd_dtw(other, ref)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """final/index.csv as C10.2 writes it (A far from R, B close) and an eval.md to extend."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    final = evaluate.final_dir()
    rows = []
    for i, pid in enumerate(["P01", "P02", "P03"]):
        waves = {"A_baseline": _tone((1500 + 100 * i, 3100)), "B_boinga": _tone((220, 660), lead=0.05),
                 "R_real": _tone((220, 660))}
        for name, wav in waves.items():
            if pid == "P03" and name == "R_real":
                continue  # a prompt without a reference is left out
            path = _write(final / name / f"{pid}.wav", wav)
            rows.append({"id": pid, "set": name, "path": str(path), "duration_s": 1.0,
                         "rms_db": -20.0, "text": "লেখা"})
    pd.DataFrame(rows).to_csv(final / "index.csv", index=False)
    report = paths.REPO / evaluate.EVAL_REL
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Blind listening test\n\n## Limitations\n\n- one rater\n", encoding="utf-8")
    return report


def test_mcd_table_appends_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    report = _setup(tmp_path, monkeypatch)
    table = eval_mcd.mcd_table()
    assert list(table["prompt_id"]) == ["P01", "P02"]
    assert (table["B_boinga"] < table["A_baseline"]).all()
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Blind listening test") and "- one rater" in text
    assert text.count(eval_mcd.SECTION) == 1 and "| P02 |" in text and "P03" not in text
    assert "lower is closer" in text and "2 prompts" in text
    assert f"=== BEGIN {evaluate.EVAL_REL} ===" in capsys.readouterr().out
    eval_mcd.mcd_table()  # re-run replaces the section instead of adding a second one
    assert report.read_text(encoding="utf-8") == text
    report.write_text(text + "\n## Later section\n\nkept\n", encoding="utf-8")
    eval_mcd.mcd_table()  # content after the section survives a re-run
    again = report.read_text(encoding="utf-8")
    assert again.count(eval_mcd.SECTION) == 1 and again.endswith("## Later section\n\nkept\n")
    (evaluate.final_dir() / "B_boinga" / "P02.wav").unlink()
    index = evaluate.final_dir() / "index.csv"
    rows = pd.read_csv(index)
    rows[~((rows["id"] == "P02") & (rows["set"] == "B_boinga"))].to_csv(index, index=False)
    with pytest.raises(KeyError, match="no clip for: P02 B_boinga"):
        eval_mcd.mcd_table()


def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
             capsys: pytest.CaptureFixture[str]) -> None:
    report = _setup(tmp_path, monkeypatch)
    assert cli.main(["eval-mcd"]) == 0
    assert "PASS: mcd table (2 prompts)" in capsys.readouterr().out
    report.unlink()
    assert cli.main(["eval-mcd"]) == 1
    assert "FAIL: eval-mcd: FileNotFoundError" in capsys.readouterr().out
