"""Tests for prepare.build() and `boinga prepare` end to end on a tiny fake corpus: skip, restore."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import cli, corpus, paths, prepare, speaker_select, text_frontend

REAL_MAP = paths.configs_dir() / text_frontend.CHAR_MAP_FILE  # resolved before any monkeypatch
TARGET = 22050
N_TRAIN, N_TEST = 50, 22  # official splits; 1 more train clip is silent (transform fails)
SILENT = "999"


def _word(cid: str) -> str:
    """A unique Bengali word per id with no digits (char_map excludes rows with digits)."""
    return "".join("কখগঘঙচছজঝঞ"[int(d)] for d in cid)


def _wav(path: Path, seconds: float, amp: float) -> str:
    t = np.arange(int(16000 * seconds)) / 16000
    y = np.concatenate([np.zeros(1600), amp * np.sin(2 * np.pi * 220 * t), np.zeros(4000)])
    sf.write(path, y.astype("float32"), 16000, subtype="PCM_16")
    return str(path)


@pytest.fixture
def fake_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pd.DataFrame]:
    """Fake repo + Drive + fast disk: selection, configs and 16 kHz wavs with Bengali text."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    cfg = paths.configs_dir()
    cfg.mkdir(parents=True)
    (cfg / text_frontend.CHAR_MAP_FILE).write_bytes(REAL_MAP.read_bytes())
    text_frontend.default_char_map.cache_clear()
    (cfg / "pretrained_facts.json").write_text(json.dumps({
        "male": {"sample_rate": TARGET},
        "dataset": {"formatter": "ljspeech", "meta_files": {
            "train": "metadata_train.csv", "val": "metadata_val.csv",
            "test": "metadata_test.csv"}}}), encoding="utf-8")
    src = tmp_path / "corpus"
    src.mkdir()
    rows = []
    for i in range(N_TRAIN + N_TEST):
        cid = str(i + 2)
        rows.append({"clip_id": cid, "wav_path": _wav(src / f"{cid}.wav", 0.4 + i % 3 * 0.1, 0.3),
                     "ctg_text": f"বাক্য {_word(cid)}।", "split_official":
                     "train" if i < N_TRAIN else "test"})
    sf.write(src / f"{SILENT}.wav", np.zeros(8000, "float32"), 16000, subtype="PCM_16")
    rows.append({"clip_id": SILENT, "wav_path": str(src / f"{SILENT}.wav"),
                 "ctg_text": "চুপ", "split_official": "train"})
    index = pd.DataFrame(rows).assign(ctg_text_nfc=lambda d: d["ctg_text"])
    sel = index[["clip_id", "split_official"]].assign(duration_s=2.0)
    data = sel.to_csv(index=False, lineterminator="\n").encode("utf-8")
    paths.speaker_selection().parent.mkdir(parents=True)
    paths.speaker_selection().write_bytes(data)
    (paths.REPO / speaker_select.SELECTION_REL).write_text(json.dumps(
        {"base_voice": "male", "csv_sha256": hashlib.sha256(data).hexdigest()}),
        encoding="utf-8")
    monkeypatch.setattr(corpus, "load_index", lambda: index)
    yield index
    text_frontend.default_char_map.cache_clear()


def _no_transform(*_: object) -> dict:
    raise AssertionError("transform_audio must not run")


def test_build_end_to_end(fake_corpus: pd.DataFrame, capsys: pytest.CaptureFixture) -> None:
    out = prepare.build(workers=1)
    printed = capsys.readouterr().out
    n_ok = N_TRAIN + N_TEST  # the silent clip fails and is left out of the manifests
    assert out["clips"] == n_ok and out["failed"] == 1
    assert f"PASS: prepared {n_ok} clips" in printed
    assert "=== BEGIN configs/splits_summary.json ===" in printed
    assert "upsampled: 72" in printed and "mean trimmed" in printed
    root = paths.data_prepared()
    lines = {k: (root / f"metadata_{k}.csv").read_text(encoding="utf-8").splitlines()
             for k in ("train", "val", "test")}
    assert len(lines["test"]) == N_TEST
    assert len(lines["val"]) in (19, 20)  # 20 sampled; the silent clip may be one of them
    assert sum(map(len, lines.values())) == n_ok
    assert not any(line.startswith(f"{SILENT}|") for v in lines.values() for line in v)
    assert lines["test"][0].split("|")[1].endswith(".")  # danda -> "." (char_map)
    for k in ("train", "val", "test"):
        assert lines[k][0] in printed  # 3 manifest lines per split are shown
    info = sf.info(str(prepare.manifest_wav(root, lines["train"][0].split("|")[0])))
    assert (info.samplerate, info.channels, info.subtype) == (TARGET, 1, "PCM_16")
    assert paths.splits_csv().is_file()
    assert len(list(paths.prepared_backup().glob("prepared-*.tar"))) == 1
    stats = pd.read_csv(root / prepare.STATS_CSV, dtype={"clip_id": str}, keep_default_na=False)
    assert stats.set_index("clip_id").at[SILENT, "error"].startswith("ValueError")


def test_second_run_skips_done_clips(fake_corpus: pd.DataFrame,
                                     monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture) -> None:
    prepare.build(workers=1)
    tar = next(paths.prepared_backup().glob("prepared-*.tar"))
    mtime = tar.stat().st_mtime_ns
    monkeypatch.setattr(prepare, "transform_audio", _no_transform)
    assert prepare.build(workers=1)["clips"] == N_TRAIN + N_TEST
    assert "PASS: prepared" in capsys.readouterr().out
    assert tar.stat().st_mtime_ns == mtime  # same splits sha: the tar is not rewritten


def test_lost_wav_is_redone(fake_corpus: pd.DataFrame) -> None:
    prepare.build(workers=1)
    wav = next((paths.data_prepared() / "wavs").glob("*.wav"))
    wav.unlink()
    prepare.build(workers=1)
    assert wav.is_file()


def test_fresh_vm_restores_from_the_drive_tar(fake_corpus: pd.DataFrame,
                                              monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
    prepare.build(workers=1)
    before = (paths.data_prepared() / "metadata_train.csv").read_bytes()
    shutil.rmtree(paths.fast())  # a new Colab VM: the fast disk is empty, Drive is not
    capsys.readouterr()
    monkeypatch.setattr(prepare, "transform_audio", _no_transform)
    out = prepare.build(workers=1)
    printed = capsys.readouterr().out
    assert "restored" in printed and out["clips"] == N_TRAIN + N_TEST
    assert (paths.data_prepared() / "metadata_train.csv").read_bytes() == before
    assert len(list((paths.data_prepared() / "wavs").glob("*.wav"))) == N_TRAIN + N_TEST


def test_build_with_a_process_pool(fake_corpus: pd.DataFrame) -> None:
    assert prepare.build(workers=2)["clips"] == N_TRAIN + N_TEST


def test_pick_prepared_reads_the_train_manifest(fake_corpus: pd.DataFrame) -> None:
    prepare.build(workers=1)
    wavs, labels = prepare.pick_prepared(5)
    assert len(wavs) == len(labels) == 5 and all(Path(w).is_file() for w in wavs)
    assert all(lab.split(": ", 1)[1].startswith("বাক্য") for lab in labels)
    assert prepare.pick_prepared(5) == (wavs, labels)  # seeded


def test_cli_prepare(fake_corpus: pd.DataFrame, capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["prepare", "--workers", "1"]) == 0
    assert "PASS: prepared" in capsys.readouterr().out
    paths.speaker_selection().write_bytes(b"clip_id,split_official,duration_s\n")
    assert cli.main(["prepare", "--workers", "1"]) == 1
    assert "FAIL: prepare: ValueError" in capsys.readouterr().out
