"""Tests for speaker_select.cluster()/candidates() and `boinga cluster-speakers` on fake voices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boinga_translator import cli, data_audit, paths, speaker_select
from test_audit_summary import _rows  # clean clips c0..c7, 3.0 s each (0.05 min)

# Unit "voices" in 4-D: A is orthogonal to B and C; C sits at cosine distance 0.32 from B,
# so thresholds 0.20-0.30 keep B and C apart and 0.35-0.50 merge them.
A = np.array([0.0, 0.0, 1.0, 0.0])
B = np.array([1.0, 0.0, 0.0, 0.0])
C = np.array([0.68, np.sqrt(1 - 0.68**2), 0.0, 0.0])
TWO = {"c0": A, "c1": A, "c2": A, "c3": B, "c4": B, "c5": B, "c6": B, "c7": B}
THREE = {"c0": A, "c1": A, "c2": A, "c3": B, "c4": B, "c5": B, "c6": C, "c7": C}


@pytest.fixture
def audit_df(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    out = paths.drive_reports() / data_audit.AUDIT_CSV
    out.parent.mkdir(parents=True)
    pd.DataFrame(_rows(), columns=list(data_audit.AUDIT_COLS)).to_csv(out, index=False)
    return data_audit.read_audit(out)


def _write(voices: dict[str, np.ndarray], emb_dir: Path) -> Path:
    emb_dir.mkdir(parents=True, exist_ok=True)
    (emb_dir / speaker_select.ENCODER_FILE).write_text("fake\n", encoding="utf-8")
    for i, (cid, v) in enumerate(voices.items()):
        jitter = np.array([0.0, 0.0, 0.0, 0.01 * i])  # same voice, not identical vectors
        np.save(emb_dir / f"{cid}.npy", (v + jitter).astype(np.float32))
    return emb_dir


def _groups(labels: pd.DataFrame) -> dict[int, list[str]]:
    return {int(k): sorted(g["clip_id"]) for k, g in labels.groupby("cluster")}


def test_two_voices_give_two_clusters_ranked_by_minutes(
        audit_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    emb = _write(TWO, paths.speaker_emb())
    labels = speaker_select.cluster(emb, audit_df, min_minutes=0.1)
    assert _groups(labels) == {1: ["c3", "c4", "c5", "c6", "c7"], 2: ["c0", "c1", "c2"]}
    assert list(labels.columns) == ["clip_id", "split_official", "duration_s", "cluster",
                                    "threshold"]
    assert labels["threshold"].eq(0.20).all()  # every threshold ties: the lowest wins
    saved = pd.read_csv(paths.speaker_clusters(), dtype={"clip_id": str})
    assert _groups(saved) == _groups(labels)
    out = capsys.readouterr().out
    assert "PASS: ranked 2 clusters" in out
    assert "=== BEGIN reports/speaker_ranking.md ===" in out
    assert speaker_select.candidates() == [1, 2]
    assert speaker_select.candidates(n=1) == [1]


def test_best_silhouette_among_thresholds_with_enough_minutes(audit_df: pd.DataFrame) -> None:
    emb = _write(THREE, paths.speaker_emb())
    labels = speaker_select.cluster(emb, audit_df, min_minutes=0.1)
    assert labels["threshold"].iloc[0] == 0.20  # B and C apart: higher silhouette
    assert _groups(labels) == {1: ["c0", "c1", "c2"], 2: ["c3", "c4", "c5"], 3: ["c6", "c7"]}


def test_choose_rule_on_a_hand_made_table() -> None:
    table = pd.DataFrame({"threshold": [0.20, 0.25, 0.30, 0.40],
                          "silhouette": [0.9, np.nan, 0.7, 0.8],
                          "largest_min": [5.0, 50.0, 40.0, 40.0]})
    # 0.20 has the best silhouette but too few minutes; 0.25 has no silhouette.
    assert speaker_select._choose(table, min_minutes=30.0) == 0.40
    assert speaker_select._choose(table, min_minutes=60.0) == 0.25  # none qualify: most min
    tie = table.assign(silhouette=[0.9, 0.8, 0.8, 0.8])
    assert speaker_select._choose(tie, min_minutes=30.0) == 0.25  # 3-way tie: lowest wins


def test_zero_embedding_is_named(audit_df: pd.DataFrame) -> None:
    emb = _write(TWO, paths.speaker_emb())
    np.save(emb / "c4.npy", np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match=r"zero or non-finite embeddings for clips \['c4'\]"):
        speaker_select.cluster(emb, audit_df, min_minutes=0.1)


def test_no_threshold_has_enough_minutes_so_most_minutes_wins(audit_df: pd.DataFrame) -> None:
    emb = _write(THREE, paths.speaker_emb())
    labels = speaker_select.cluster(emb, audit_df, min_minutes=30.0)
    assert labels["threshold"].iloc[0] == 0.35  # B+C merged: 5 clips, the biggest cluster
    assert _groups(labels)[1] == ["c3", "c4", "c5", "c6", "c7"]


def test_report_tables_and_sample_ids(audit_df: pd.DataFrame) -> None:
    emb = _write(THREE, paths.speaker_emb())
    speaker_select.cluster(emb, audit_df, min_minutes=0.1)
    text = (paths.REPO / speaker_select.RANKING_REL).read_text(encoding="utf-8")
    assert "| threshold | clusters | largest cluster min | silhouette | chosen |" in text
    assert text.count("| 0.") >= len(speaker_select.THRESHOLDS)
    assert "| 1 | 3 | 0.15 |" in text  # rank, clips, clean minutes
    assert "c6, c7" in text  # up to 5 clip ids per cluster, for listening
    assert "Encoder: fake" in text


def test_only_clean_clips_with_an_embedding_count(audit_df: pd.DataFrame) -> None:
    voices = {k: v for k, v in TWO.items() if k != "c7"}
    emb = _write(voices, paths.speaker_emb())
    np.save(emb / "short.npy", B.astype(np.float32))  # "short" is not a clean clip
    labels = speaker_select.cluster(emb, audit_df, min_minutes=0.1)
    assert sorted(labels["clip_id"]) == sorted(voices)


def test_too_few_embeddings_raise(audit_df: pd.DataFrame) -> None:
    emb = _write({"c0": A}, paths.speaker_emb())
    with pytest.raises(ValueError, match="at least 3"):
        speaker_select.cluster(emb, audit_df)


def test_cli_cluster_speakers(audit_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["cluster-speakers"]) == 1  # no embeddings yet
    assert "FAIL: cluster-speakers" in capsys.readouterr().out
    _write(TWO, paths.speaker_emb())
    assert cli.main(["cluster-speakers"]) == 0
    assert "PASS: ranked 2 clusters" in capsys.readouterr().out
