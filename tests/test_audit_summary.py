"""Tests for audit_summary.py and `boinga audit-summary`: exclusion rules, report, samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from boinga_translator import audit_summary, cli, data_audit, paths

REASONS = ["undecodable", "too_short", "too_long", "clipped", "quiet", "empty_text", "oov",
           "speech_rate", "duplicate_text"]


def _row(cid: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {c: "" for c in data_audit.AUDIT_COLS}
    base.update(clip_id=cid, split_official="train", wav_path=f"/w/{cid}.wav",
                ctg_text_nfc=f"লেখা {cid}", charset_id="female-abc", decodable=True,
                sample_rate=16000, channels=1, subtype="PCM_16", duration_s=3.0, peak=0.9,
                clip_ratio=0.0, rms_db=-20.0, lead_silence_s=0.1, trail_silence_s=0.1,
                n_chars=20, n_words=4, empty=False, oov_ratio=0.0, has_latin=False,
                has_digits=False, chars_per_second=12.0, duplicate_text=False)
    base.update(kw)
    return base


def _rows() -> list[dict[str, Any]]:
    clean = [_row(f"c{i}", chars_per_second=10.0 + i, split_official=s)
             for i, s in enumerate(["train"] * 5 + ["val", "test", "test"])]
    return [*clean,
            _row("undec", decodable=False, error="no wav file", duration_s="", split_official="",
                 chars_per_second=""),
            _row("short", duration_s=0.5), _row("long", duration_s=16.0),
            _row("clip", clip_ratio=0.01), _row("quiet", rms_db=-50.0),
            _row("empty", empty=True), _row("oov", oov_ratio=0.05, chars_not_in_model="।"),
            _row("oov2", oov_ratio=0.05, chars_not_in_model="।ৰ"),
            _row("slow", chars_per_second=1.0), _row("fast", chars_per_second=40.0),
            _row("dup1", duplicate_text=True), _row("dup2", duplicate_text=True)]


@pytest.fixture
def audit_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    out = paths.drive_reports() / "audit.csv"
    out.parent.mkdir(parents=True)
    pd.DataFrame(_rows(), columns=list(data_audit.AUDIT_COLS)).to_csv(out, index=False)
    return out


def test_exclusion_reasons_and_clean_mask(audit_csv: Path) -> None:
    df = data_audit.read_audit(audit_csv)
    reasons = audit_summary.exclusion_reasons(df)
    assert list(reasons.columns) == REASONS
    flagged = {r: sorted(df.loc[reasons[r], "clip_id"]) for r in REASONS}
    assert flagged["undecodable"] == ["undec"]
    assert flagged["too_short"] == ["short"] and flagged["too_long"] == ["long"]
    assert flagged["clipped"] == ["clip"] and flagged["quiet"] == ["quiet"]
    assert flagged["empty_text"] == ["empty"] and flagged["oov"] == ["oov", "oov2"]
    assert flagged["duplicate_text"] == ["dup1", "dup2"]
    assert flagged["speech_rate"] == ["fast", "slow", "undec"]  # outside 1st-99th pct or NaN
    clean = audit_summary.clean_mask(df)
    assert sorted(df.loc[clean, "clip_id"]) == [f"c{i}" for i in range(8)]


def test_summarise_writes_and_emits_report(audit_csv: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    text = audit_summary.summarise(audit_csv)
    out = capsys.readouterr().out
    assert "=== BEGIN reports/audit_summary.md ===" in out
    assert out.rstrip().endswith("PASS: summary written")
    assert (paths.REPO / "reports" / "audit_summary.md").read_text(encoding="utf-8") == text
    assert "| train |" in text and "| val |" in text and "| (none) |" in text
    assert "| 16000 | 1 | PCM_16 |" in text
    assert "| oov (oov_ratio > 0.02) | 2 |" in text
    danda_part = text.split("oov only because of the danda")[1].split("Clean:")[0]
    assert danda_part.strip().splitlines()[-1] == "| 1 |"  # "oov" yes, "oov2" (ৰ too) no
    assert "Clean: 8 clips" in text
    assert "no speaker field" in text
    assert "| । | U+0964 | DEVANAGARI DANDA | 2 |" in text


def test_speaker_table_sorted_by_clean_minutes(audit_csv: Path) -> None:
    df = pd.read_csv(audit_csv, dtype=str, keep_default_na=False)
    df["speaker_id"] = ["A", "B", "B", "B", "A", "A", "A", "A"] + ["A"] * (len(df) - 8)
    df.to_csv(audit_csv, index=False)
    text = audit_summary.summarise(audit_csv)
    table = text.split("## Speakers")[1]
    assert table.index("| A |") < table.index("| B |")  # A has 5 clean clips, B has 3


def test_sample_clips_are_clean_seeded_and_labelled(audit_csv: Path) -> None:
    index = pd.DataFrame({"clip_id": [f"c{i}" for i in range(8)],
                          "ctg_text": [f"চাটগাঁইয়া {i}" for i in range(8)],
                          "bn_text": [f"বাংলা {i}" for i in range(8)]})
    wavs, labels = audit_summary.sample_clips(4, seed=7, audit_csv=audit_csv, index_df=index)
    again = audit_summary.sample_clips(4, seed=7, audit_csv=audit_csv, index_df=index)
    assert (wavs, labels) == again and len(wavs) == len(labels) == 4
    ids = [Path(w).stem for w in wavs]
    assert set(ids) <= {f"c{i}" for i in range(8)}
    first = labels[0]
    n = ids[0][1:]
    assert first.startswith(f"{ids[0]} [") and f"CTG: চাটগাঁইয়া {n}" in first
    assert f"BN: বাংলা {n}" in first
    assert len(audit_summary.sample_clips(50, audit_csv=audit_csv, index_df=index)[0]) == 8


def test_empty_audit_and_blank_bool(audit_csv: Path) -> None:
    df = pd.read_csv(audit_csv, dtype=str, keep_default_na=False)
    df.loc[0, "decodable"] = ""  # blank bool -> False (undecodable), no crash
    df.to_csv(audit_csv, index=False)
    typed = data_audit.read_audit(audit_csv)
    assert typed["decodable"].dtype == bool and not typed.loc[0, "decodable"]
    assert audit_summary.exclusion_reasons(typed).loc[0, "undecodable"]
    df.iloc[:0].to_csv(audit_csv, index=False)
    with pytest.raises(ValueError, match="no rows"):
        audit_summary.summarise(audit_csv)


def test_cli_audit_summary(audit_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["audit-summary"]) == 0
    assert "PASS: summary written" in capsys.readouterr().out
    audit_csv.unlink()
    assert cli.main(["audit-summary"]) == 1
    assert "FAIL: audit-summary:" in capsys.readouterr().out
