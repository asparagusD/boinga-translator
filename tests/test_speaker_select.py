"""Tests for speaker_select.embed() and `boinga embed-speakers`: clean clips only, per-clip cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boinga_translator import cli, data_audit, paths, speaker_select
from test_audit_summary import _rows  # same synthetic audit rows: clean clips are c0..c7

CLEAN = [f"c{i}" for i in range(8)]


@pytest.fixture
def audit_df(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    out = paths.drive_reports() / data_audit.AUDIT_CSV
    out.parent.mkdir(parents=True)
    pd.DataFrame(_rows(), columns=list(data_audit.AUDIT_COLS)).to_csv(out, index=False)
    return data_audit.read_audit(out)


def _fake(name: str = "fake", fail: set[str] | None = None) -> tuple[speaker_select.Encoder,
                                                                      list[str]]:
    """Encoder that records the wav paths it sees; vector = clip number, 4 dims."""
    seen: list[str] = []

    def embed(wav_path: str) -> np.ndarray:
        cid = Path(wav_path).stem
        seen.append(cid)
        if cid in (fail or set()):
            raise ValueError("no speech left after trimming")
        return np.full(4, float(cid.lstrip("c")))

    return speaker_select.Encoder(name, embed), seen


def _no_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> speaker_select.Encoder:
        raise AssertionError("encoder loaded although every clip was cached")
    monkeypatch.setattr(speaker_select, "load_encoder", boom)


def test_embeds_only_clean_clips_one_npy_each(audit_df: pd.DataFrame, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    enc, seen = _fake()
    cache = tmp_path / "emb"
    out = speaker_select.embed(audit_df, cache, encoder=enc)
    assert sorted(seen) == CLEAN
    assert sorted(p.stem for p in cache.glob("*.npy")) == CLEAN
    np.testing.assert_array_equal(np.load(cache / "c3.npy"), np.full(4, 3.0, dtype=np.float32))
    assert list(out.columns) == ["clip_id", "split_official", "npy"]
    assert sorted(out["clip_id"]) == CLEAN
    assert (cache / speaker_select.ENCODER_FILE).read_text(encoding="utf-8").strip() == "fake"
    assert "PASS: 8 embeddings (fake, 0 failed)" in capsys.readouterr().out
    assert not list(cache.glob("*.tmp"))


def test_cached_clips_are_skipped(audit_df: pd.DataFrame, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    cache = tmp_path / "emb"
    speaker_select.embed(audit_df, cache, encoder=_fake()[0])
    _no_loader(monkeypatch)
    out = speaker_select.embed(audit_df, cache)  # all cached: no encoder is loaded
    assert len(out) == 8
    assert "PASS: 8 embeddings (fake, 0 failed)" in capsys.readouterr().out
    (cache / "c5.npy").unlink()
    enc, seen = _fake()
    speaker_select.embed(audit_df, cache, encoder=enc)
    assert seen == ["c5"]


def test_cache_from_another_encoder_is_refused(audit_df: pd.DataFrame, tmp_path: Path) -> None:
    cache = tmp_path / "emb"
    speaker_select.embed(audit_df, cache, encoder=_fake("resemblyzer-0.1.4")[0])
    (cache / "c0.npy").unlink()
    with pytest.raises(RuntimeError, match="resemblyzer-0.1.4"):
        speaker_select.embed(audit_df, cache, encoder=_fake("speechbrain-ecapa")[0])


def test_npy_from_an_unknown_encoder_is_refused(audit_df: pd.DataFrame, tmp_path: Path) -> None:
    cache = tmp_path / "emb"
    speaker_select.embed(audit_df, cache, encoder=_fake()[0])
    (cache / speaker_select.ENCODER_FILE).unlink()
    (cache / "c0.npy").unlink()
    with pytest.raises(RuntimeError, match="no encoder.txt"):
        speaker_select.embed(audit_df, cache, encoder=_fake()[0])


def test_unsafe_clip_id_is_refused(audit_df: pd.DataFrame, tmp_path: Path) -> None:
    bad = audit_df.assign(clip_id=audit_df["clip_id"].replace("c0", "../c0"))
    with pytest.raises(ValueError, match="not a safe file name"):
        speaker_select.embed(bad, tmp_path / "emb", encoder=_fake()[0])


def test_too_many_failures_stop_early_and_keep_the_rest(audit_df: pd.DataFrame,
                                                        tmp_path: Path) -> None:
    cache = tmp_path / "emb"
    enc, seen = _fake(fail={"c2"})
    with pytest.raises(RuntimeError, match="1 of 8 clips failed"):
        speaker_select.embed(audit_df, cache, encoder=enc)
    assert seen == ["c0", "c1", "c2"]  # 1 of 8 is over 1%: stopped at the first failure
    assert sorted(p.stem for p in cache.glob("*.npy")) == ["c0", "c1"]  # resumable
    assert "c2" in (cache / speaker_select.ERRORS_FILE).read_text(encoding="utf-8")


def test_load_encoder_falls_back_to_speechbrain(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    def no_resemblyzer() -> speaker_select.Encoder:
        raise ModuleNotFoundError("No module named 'pkg_resources'")

    monkeypatch.setattr(speaker_select, "_resemblyzer", no_resemblyzer)
    monkeypatch.setattr(speaker_select, "_speechbrain", lambda: _fake("sb")[0])
    assert speaker_select.load_encoder().name == "sb"
    assert "WARN: resemblyzer unavailable" in capsys.readouterr().out

    def no_speechbrain() -> speaker_select.Encoder:
        raise ModuleNotFoundError("No module named 'speechbrain'")

    monkeypatch.setattr(speaker_select, "_speechbrain", no_speechbrain)
    with pytest.raises(RuntimeError, match=r"speaker-fallback"):
        speaker_select.load_encoder()


def test_cli_embed_speakers(audit_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(speaker_select, "load_encoder", lambda: _fake()[0])
    assert cli.main(["embed-speakers"]) == 0
    assert "PASS: 8 embeddings" in capsys.readouterr().out
    assert len(list(paths.speaker_emb().glob("*.npy"))) == 8
    (paths.drive_reports() / data_audit.AUDIT_CSV).unlink()
    assert cli.main(["embed-speakers"]) == 1
    assert "FAIL: embed-speakers: FileNotFoundError" in capsys.readouterr().out
