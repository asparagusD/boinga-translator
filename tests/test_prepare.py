"""Tests for prepare.py: target rate, transform_audio on synthetic wavs, leak-free splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from boinga_translator import paths, prepare, text_frontend

TARGET = 22050
PEAK = 10 ** (prepare.PEAK_DBFS / 20)
LSB = 1 / 32768  # one PCM_16 step
KEYS = {"in_sr", "out_sr", "in_dur", "out_dur", "trimmed_s", "gain_db", "upsampled"}


def _sine(seconds: float, amp: float, sr: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype("float32")


def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * seconds), "float32")


def _write(path: Path, data: np.ndarray, sr: int) -> Path:
    sf.write(path, data, sr, subtype="PCM_16")
    return path


def _lead_zeros(y: np.ndarray) -> int:
    return int(np.argmax(np.abs(y) > 1e-3))


# --- target_sample_rate ----------------------------------------------------------------------


def _configs(tmp_path: Path, voice: str, rate: object) -> tuple[Path, Path]:
    sel = tmp_path / "speaker_selection.json"
    sel.write_text(json.dumps({"base_voice": voice}), encoding="utf-8")
    facts = tmp_path / "pretrained_facts.json"
    facts.write_text(json.dumps({"male": {"sample_rate": rate},
                                 "female": {"sample_rate": 16000}}), encoding="utf-8")
    return sel, facts


def test_target_sample_rate_follows_the_base_voice(tmp_path: Path) -> None:
    sel, facts = _configs(tmp_path, "male", 24000)
    assert prepare.target_sample_rate(sel, facts) == 24000
    sel, facts = _configs(tmp_path, "female", 24000)
    assert prepare.target_sample_rate(sel, facts) == 16000


def test_target_sample_rate_of_the_repo_configs() -> None:
    assert prepare.target_sample_rate() == TARGET  # male base voice, pretrained config


@pytest.mark.parametrize(("voice", "rate"), [("robot", 22050), ("male", None), ("male", 0),
                                             ("male", "22050")])
def test_target_sample_rate_refuses_bad_configs(tmp_path: Path, voice: str, rate: object) -> None:
    sel, facts = _configs(tmp_path, voice, rate)
    with pytest.raises(ValueError):
        prepare.target_sample_rate(sel, facts)


# --- transform_audio -------------------------------------------------------------------------


def test_stereo_is_mixed_to_mono(tmp_path: Path) -> None:
    left = _sine(1.0, 0.5, TARGET)
    src = _write(tmp_path / "st.wav", np.stack([left, 0.5 * left], axis=1), TARGET)
    out = tmp_path / "out" / "st.wav"
    s = prepare.transform_audio(src, out, TARGET)
    assert set(s) == KEYS
    info = sf.info(str(out))
    assert (info.channels, info.samplerate, info.subtype) == (1, TARGET, "PCM_16")
    y, _ = sf.read(str(out), dtype="float32")
    # the mix is 0.75 * left, then peak-normalised: same shape as left
    assert np.corrcoef(y[: left.size], left[: y.size])[0, 1] > 0.999


def test_48k_is_resampled_to_the_target(tmp_path: Path) -> None:
    src = _write(tmp_path / "hi.wav", _sine(1.0, 0.5, 48000), 48000)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert (s["in_sr"], s["out_sr"], s["upsampled"]) == (48000, TARGET, False)
    assert sf.info(str(tmp_path / "o.wav")).samplerate == TARGET
    assert s["in_dur"] == pytest.approx(1.0)
    assert s["out_dur"] == pytest.approx(1.0, abs=0.01)


def test_8k_is_flagged_upsampled(tmp_path: Path) -> None:
    src = _write(tmp_path / "lo.wav", _sine(1.0, 0.5, 8000), 8000)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert (s["in_sr"], s["out_sr"], s["upsampled"]) == (8000, TARGET, True)
    assert sf.info(str(tmp_path / "o.wav")).samplerate == TARGET


def test_at_target_rate_is_not_upsampled(tmp_path: Path) -> None:
    src = _write(tmp_path / "eq.wav", _sine(1.0, 0.5, TARGET), TARGET)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert s["upsampled"] is False and s["in_sr"] == s["out_sr"] == TARGET


def test_edge_silence_is_trimmed_to_a_100ms_pad(tmp_path: Path) -> None:
    y = np.concatenate([_silence(0.5, TARGET), _sine(1.0, 0.5, TARGET), _silence(0.5, TARGET)])
    src = _write(tmp_path / "pad.wav", y, TARGET)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    out, _ = sf.read(str(tmp_path / "o.wav"), dtype="float32")
    pad = int(prepare.PAD_S * TARGET)
    extra = 1024 + 512  # trim's centred frames keep up to ~half a frame (+ one hop) more
    assert pad <= _lead_zeros(out) <= pad + extra  # never less than the 100 ms pad
    assert pad <= _lead_zeros(out[::-1]) <= pad + extra
    assert s["in_dur"] == pytest.approx(2.0)
    assert 1.2 <= s["out_dur"] <= 1.2 + 2 * extra / TARGET
    assert s["trimmed_s"] == pytest.approx(s["in_dur"] - s["out_dur"], abs=1e-6)


@pytest.mark.parametrize("lead", [True, False])
def test_pad_is_clamped_at_the_clip_edges(tmp_path: Path, lead: bool) -> None:
    parts = [_silence(0.03, TARGET), _sine(1.0, 0.5, TARGET)]
    y = np.concatenate(parts if lead else parts[::-1])
    src = _write(tmp_path / "edge.wav", y, TARGET)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert s["trimmed_s"] == 0.0  # less than 100 ms of silence: nothing cut, nothing added
    assert s["out_dur"] == pytest.approx(s["in_dur"])


@pytest.mark.parametrize("amp", [0.05, 0.5, 1.0])
def test_peak_is_normalised_to_minus_1_dbfs(tmp_path: Path, amp: float) -> None:
    src = _write(tmp_path / "a.wav", _sine(1.0, amp, 16000), 16000)
    s = prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    out, _ = sf.read(str(tmp_path / "o.wav"), dtype="float32")
    peak = float(np.abs(out).max())
    assert peak <= PEAK + LSB  # nothing above -1 dBFS
    assert peak == pytest.approx(PEAK, abs=0.01)
    assert (s["gain_db"] > 0) == (amp < PEAK)


def test_silent_clip_is_refused_and_not_written(tmp_path: Path) -> None:
    src = _write(tmp_path / "z.wav", _silence(1.0, 16000), 16000)
    with pytest.raises(ValueError, match="silent"):
        prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert not (tmp_path / "o.wav").exists()


def test_non_finite_clip_is_refused_and_not_written(tmp_path: Path) -> None:
    y = _sine(1.0, 0.5, 16000)
    y[100] = np.nan
    src = tmp_path / "nan.wav"
    sf.write(src, y, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        prepare.transform_audio(src, tmp_path / "o.wav", TARGET)
    assert not (tmp_path / "o.wav").exists()


def test_empty_clip_is_refused(tmp_path: Path) -> None:
    src = _write(tmp_path / "e.wav", np.zeros(0, "float32"), 16000)
    with pytest.raises(ValueError, match="empty"):
        prepare.transform_audio(src, tmp_path / "o.wav", TARGET)


# --- make_splits / save_splits ---------------------------------------------------------------


def _index(n_train: int, n_val: int, n_test: int, dupes: int = 0) -> pd.DataFrame:
    """Unique texts per clip; `dupes` extra clips repeat train texts (longer, then shorter)."""
    rows = []
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(n):
            i = len(rows)
            rows.append({"clip_id": str(i + 2), "ctg_text_nfc": f"text {i}",
                         "split_official": split, "duration_s": 2.0 + (i % 7)})
    for k in range(dupes):  # copy of train row k: +1 s if k is even (kept), -1 s if odd
        src = rows[k]
        rows.append({**src, "clip_id": str(len(rows) + 2),
                     "duration_s": src["duration_s"] + (1.0 if k % 2 == 0 else -1.0)})
    return pd.DataFrame(rows)


def _check_no_leak(index: pd.DataFrame, splits: pd.DataFrame) -> None:
    text = index.set_index("clip_id")["ctg_text_nfc"]
    per_text = splits.assign(t=splits["clip_id"].map(text)).groupby("t")["split"].nunique()
    assert (per_text == 1).all()
    assert not splits["clip_id"].duplicated().any()


def test_make_splits_uses_the_official_test_when_big_enough() -> None:
    idx = _index(400, 60, 30)
    sp = prepare.make_splits(idx, idx["clip_id"])
    assert list(sp.columns) == ["clip_id", "split", "duration_s"]
    test_ids = set(idx.loc[idx["split_official"] == "test", "clip_id"])
    assert set(sp.loc[sp["split"] == "test", "clip_id"]) == test_ids
    assert sp.attrs["test_source"] == "official"
    assert (sp["split"] == "val").sum() == 23  # 5% of the 460 left after test
    assert (sp["split"] == "train").sum() == 460 - 23
    _check_no_leak(idx, sp)


def test_make_splits_samples_test_when_official_is_small() -> None:
    idx = _index(500, 50, 19)
    sp = prepare.make_splits(idx, idx["clip_id"])
    assert sp.attrs["test_source"] == "sampled"
    assert (sp["split"] == "test").sum() == 28  # 5% of 569 unique texts, rounded
    assert (sp["split"] == "val").sum() == 27  # 5% of the 541 left
    _check_no_leak(idx, sp)


def test_make_splits_minimum_20_per_held_out_split() -> None:
    idx = _index(100, 0, 0)
    sp = prepare.make_splits(idx, idx["clip_id"])
    counts = sp["split"].value_counts()
    assert (counts["test"], counts["val"], counts["train"]) == (20, 20, 60)


def test_make_splits_keeps_selected_clips_only() -> None:
    idx = _index(200, 0, 30)
    chosen = idx["clip_id"].iloc[::2]
    sp = prepare.make_splits(idx, chosen)
    assert set(sp["clip_id"]) == set(chosen)
    assert sp.attrs["n_selected"] == len(chosen)


def test_make_splits_dedupes_text_keeping_the_longest_clip() -> None:
    idx = _index(300, 0, 30, dupes=6)
    sp = prepare.make_splits(idx, idx["clip_id"])
    assert len(sp) == 330 and sp.attrs["dropped_duplicate_text"] == 6
    kept = set(sp["clip_id"])
    for k in range(6):
        copy = str(330 + k + 2)
        longer = copy if k % 2 == 0 else str(k + 2)
        shorter = str(k + 2) if k % 2 == 0 else copy
        assert longer in kept and shorter not in kept
    _check_no_leak(idx, sp)


def test_make_splits_counts_add_up() -> None:
    idx = _index(300, 40, 25, dupes=4)
    sp = prepare.make_splits(idx, idx["clip_id"])
    assert len(sp) + sp.attrs["dropped_duplicate_text"] == sp.attrs["n_selected"] == len(idx)
    assert set(sp["split"]) == {"train", "val", "test"}


def test_make_splits_is_deterministic_and_seeded() -> None:
    idx = _index(400, 0, 0)
    a = prepare.make_splits(idx, idx["clip_id"], seed=7)
    b = prepare.make_splits(idx.sample(frac=1, random_state=3), idx["clip_id"], seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = prepare.make_splits(idx, idx["clip_id"], seed=8)
    assert set(a.loc[a["split"] == "test", "clip_id"]) != set(c.loc[c["split"] == "test",
                                                                     "clip_id"])
    assert a.attrs["seed"] == 7


def test_make_splits_refuses_unknown_ids_and_tiny_pools() -> None:
    idx = _index(100, 0, 0)
    with pytest.raises(ValueError, match="not in the index"):
        prepare.make_splits(idx, ["nope"])
    with pytest.raises(ValueError, match="duration_s"):
        prepare.make_splits(idx.drop(columns="duration_s"), idx["clip_id"])
    small = _index(40, 0, 0)
    with pytest.raises(ValueError, match="no clips left for train"):
        prepare.make_splits(small, small["clip_id"])


def test_save_splits_writes_csv_and_summary(tmp_path: Path) -> None:
    idx = _index(300, 0, 30)
    sp = prepare.make_splits(idx, idx["clip_id"])
    csv, summ = tmp_path / "splits.csv", tmp_path / "splits_summary.json"
    out = prepare.save_splits(sp, csv, summ)
    data = json.loads(summ.read_text(encoding="utf-8"))
    assert out == data
    assert data["csv_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert (data["seed"], data["test_source"]) == (sp.attrs["seed"], "official")
    assert data["counts"] == {"train": 280, "val": 20, "test": 30}  # 5% of 300 < min 20
    want = sp.groupby("split")["duration_s"].sum() / 60
    assert data["minutes"] == {k: round(float(want[k]), 2) for k in ("train", "val", "test")}
    back = pd.read_csv(csv, dtype={"clip_id": str})
    assert list(back["clip_id"]) == list(sp["clip_id"]) and list(back["split"]) == list(sp["split"])


def test_save_splits_default_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path))
    idx = _index(100, 0, 0)
    prepare.save_splits(prepare.make_splits(idx, idx["clip_id"]), summary_path=tmp_path / "s.json")
    assert paths.splits_csv() == tmp_path / "splits.csv" and paths.splits_csv().is_file()


def test_save_splits_never_changes_a_saved_test_set(tmp_path: Path) -> None:
    idx = _index(400, 0, 0)
    csv, summ = tmp_path / "splits.csv", tmp_path / "s.json"
    prepare.save_splits(prepare.make_splits(idx, idx["clip_id"], seed=1), csv, summ)
    prepare.save_splits(prepare.make_splits(idx, idx["clip_id"], seed=1), csv, summ)  # same: ok
    before = csv.read_bytes()
    with pytest.raises(ValueError, match="test set"):
        prepare.save_splits(prepare.make_splits(idx, idx["clip_id"], seed=2), csv, summ)
    assert csv.read_bytes() == before


def test_save_splits_needs_make_splits_metadata(tmp_path: Path) -> None:
    bare = pd.DataFrame({"clip_id": ["1"], "split": ["train"], "duration_s": [1.0]})
    with pytest.raises(ValueError, match="make_splits"):
        prepare.save_splits(bare, tmp_path / "a.csv", tmp_path / "b.json")


# --- manifests ---------------------------------------------------------------------------------

NAMES = {"train": "metadata_train.csv", "val": "metadata_val.csv", "test": "metadata_test.csv"}
BN = "আঁই তোঁয়ারে ভালা পাই।"  # Chittagonian line; NFC already, the danda maps to "." (char_map)


def _facts(tmp_path: Path, dataset: object = None) -> Path:
    facts = tmp_path / "facts.json"
    ds = {"formatter": "ljspeech", "meta_files": NAMES} if dataset is None else dataset
    facts.write_text(json.dumps({"male": {"sample_rate": 22050}, "dataset": ds}),
                     encoding="utf-8")
    return facts


def _split_frames(texts: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = list(texts)
    split = ["train"] * (len(ids) - 2) + ["val", "test"]
    sp = pd.DataFrame({"clip_id": ids, "split": split, "duration_s": 2.0})
    return sp, pd.DataFrame({"clip_id": ids, "ctg_text": list(texts.values())})


def test_manifest_names_come_from_the_repo_facts() -> None:
    assert prepare.manifest_names() == NAMES


@pytest.mark.parametrize("bad", [{"formatter": "vctk", "meta_files": NAMES},
                                 {"formatter": "ljspeech", "meta_files": {"train": "a.csv"}},
                                 "nope"])
def test_manifest_names_refuse_bad_facts(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ValueError):
        prepare.manifest_names(_facts(tmp_path, bad))


def test_write_manifests_ljspeech_lines(tmp_path: Path) -> None:
    sp, idx = _split_frames({"10": BN, "11": "  এক   দুই ", "12": "তিন", "13": "চার"})
    out = prepare.write_manifests(sp, idx, tmp_path, facts_path=_facts(tmp_path))
    assert out == {k: tmp_path / v for k, v in NAMES.items()}
    raw = (tmp_path / NAMES["train"]).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert len(lines) == 2  # no header
    cols = [line.split("|") for line in lines]
    assert all(len(c) == 3 for c in cols)  # separator "|", exactly 3 columns
    want = text_frontend.prepare_text(BN)
    assert cols[0] == ["10", want, want]
    assert "আঁই" in want and want.endswith(".")  # Bengali kept, only char_map applied
    assert cols[1] == ["11", "এক দুই", "এক দুই"]  # prepare_text: whitespace collapsed
    assert (tmp_path / NAMES["val"]).read_text(encoding="utf-8") == "12|তিন|তিন\n"
    assert (tmp_path / NAMES["test"]).read_text(encoding="utf-8") == "13|চার|চার\n"


def test_wav_paths_live_under_wavs(tmp_path: Path) -> None:
    assert prepare.manifest_wav(tmp_path, "386") == tmp_path / "wavs" / "386.wav"


@pytest.mark.parametrize("text", ["a|b", "a\rb", "", "   "])
def test_write_manifests_refuses_unsafe_text(tmp_path: Path, text: str,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    # char_map turns "|" into "." and normalise() collapses CR, so bypass the front-end to
    # prove the guard itself holds if either ever changes
    monkeypatch.setattr(prepare.text_frontend, "prepare_text", lambda t: t.strip())
    sp, idx = _split_frames({"1": "এক", "2": text, "3": "দুই", "4": "তিন"})
    with pytest.raises(ValueError, match="clip 2"):
        prepare.write_manifests(sp, idx, tmp_path, facts_path=_facts(tmp_path))
    assert not list(tmp_path.glob("metadata_*.csv"))


def test_write_manifests_refuses_missing_text(tmp_path: Path) -> None:
    sp, idx = _split_frames({"1": "এক", "2": "দুই", "3": "তিন"})
    with pytest.raises(ValueError, match="no transcript"):
        prepare.write_manifests(sp, idx.iloc[1:], tmp_path, facts_path=_facts(tmp_path))
    with pytest.raises(ValueError, match="twice"):
        prepare.write_manifests(sp, pd.concat([idx, idx]), tmp_path,
                                facts_path=_facts(tmp_path))
    with pytest.raises(ValueError, match="unknown split"):
        prepare.write_manifests(sp.assign(split="dev"), idx, tmp_path,
                                facts_path=_facts(tmp_path))


# --- parse check (fake coqui: the ljspeech formatter keeps the line's "\n" on column 2) ------


def _ljspeech(root: Path, meta: str) -> list[dict[str, str]]:
    items = []
    with open(root / meta, encoding="utf-8") as f:
        for line in f:
            cols = line.split("|")
            items.append({"text": cols[2], "audio_file": str(root / "wavs" / f"{cols[0]}.wav"),
                          "root_path": str(root)})
    return items


def _fake_load(root: Path, train: str, val: str | None) -> tuple[list, list]:
    return _ljspeech(root, train), _ljspeech(root, val) if val else []


def _prepared(tmp_path: Path, wavs: bool = True) -> Path:
    sp, idx = _split_frames({"1": "এক", "2": "দুই", "3": "তিন", "4": "চার", "5": "পাঁচ"})
    facts = _facts(tmp_path)
    root = tmp_path / "prepared"
    prepare.write_manifests(sp, idx, root, facts_path=facts)
    if wavs:
        (root / "wavs").mkdir()
        for cid in sp["clip_id"]:
            _write(prepare.manifest_wav(root, cid), _sine(0.2, 0.5, TARGET), TARGET)
    return facts


def test_parse_check_passes_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    facts = _prepared(tmp_path)
    counts = prepare.parse_check(tmp_path / "prepared", loader=_fake_load, facts_path=facts)
    assert counts == {"train": 3, "val": 1, "test": 1}
    out = capsys.readouterr().out
    assert "PASS: coqui parsed 3 train / 1 val samples" in out
    assert "text='এক\\n'" in out  # first sample's text shown with repr, trailing newline visible
    assert "1.wav" in out


def test_parse_check_fails_on_a_missing_wav(tmp_path: Path) -> None:
    facts = _prepared(tmp_path)
    prepare.manifest_wav(tmp_path / "prepared", "4").unlink()
    with pytest.raises(RuntimeError, match="1 audio file"):
        prepare.parse_check(tmp_path / "prepared", loader=_fake_load, facts_path=facts)


def test_parse_check_fails_when_the_text_differs(tmp_path: Path) -> None:
    facts = _prepared(tmp_path)

    def column1(root: Path, train: str, val: str | None) -> tuple[list, list]:
        a, b = _fake_load(root, train, val)
        return [{**s, "text": s["text"] + "x"} for s in a], b

    with pytest.raises(RuntimeError, match="text differs"):
        prepare.parse_check(tmp_path / "prepared", loader=column1, facts_path=facts)


def test_parse_check_fails_on_a_count_mismatch(tmp_path: Path) -> None:
    facts = _prepared(tmp_path)

    def drop(root: Path, train: str, val: str | None) -> tuple[list, list]:
        a, b = _fake_load(root, train, val)
        return a[1:], b

    with pytest.raises(RuntimeError, match="train: parsed 2"):
        prepare.parse_check(tmp_path / "prepared", loader=drop, facts_path=facts)


def test_check_manifest_parse_tool(monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture) -> None:
    import check_manifest_parse

    monkeypatch.setattr(prepare, "parse_check", lambda: {"train": 1, "val": 1, "test": 1})
    assert check_manifest_parse.main() == 0

    def boom() -> None:
        raise RuntimeError("bad")

    monkeypatch.setattr(prepare, "parse_check", boom)
    assert check_manifest_parse.main() == 1
    assert "FAIL: manifest parse check: bad" in capsys.readouterr().out
