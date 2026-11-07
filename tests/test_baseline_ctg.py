"""Tests for baseline.pick_ctg_prompts/write_ctg_prompts and `boinga pick-baseline-ctg`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from boinga_translator import baseline, cli, corpus, data_audit, paths
from test_audit_summary import _row  # one audit row with sane defaults

BN = "আমি আজ সকালে বাজারে গিয়ে তাজা সবজি কিনেছি"


def _clip(cid: str, ctg: str, split: str = "test", **kw: Any) -> tuple[dict, dict]:
    audit = _row(cid, split_official=split, ctg_text_nfc=" ".join(ctg.split()), **kw)
    return audit, {"clip_id": cid, "ctg_text": ctg, "bn_text": BN}


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    clips = [
        _clip("10", "আঁই আজিয়া বিয়ানে বাজারত যাই তাজা তরকারি কিনগি"),  # very dialectal, 7 words
        _clip("11", " হেতে কালিয়া বিয়ালে আঁরার বাড়িত চা হাইতো আইবো "),  # spaces kept verbatim
        _clip("12", "আমি আজ সকালে বাজারে গিয়ে তাজা সবজি কিনেছি"),  # = BN: ratio 1.0
        _clip("13", "অনে কডে যাইবান আজিয়া বিয়ালে কন কামত"),
        _clip("14", "আঁই আজিয়া বাজারত যাই তরকারি কিনগি", split="train"),  # not test split
        _clip("15", "ইতারা হনো দিন এডে ন আইয়ে মনে হয়", rms_db=-60.0),  # not clean
        _clip("16", "আঁই যাই"),  # 2 words: outside 5-12
        _clip("17", "আঁই বাজারত গেইলাম তারপর হনো হাম ন গরি ঘরত আই বই রইলাম আর হানা হাইলাম "
                    "আর ঘুম গেলাম রাইত অইয়ে"),  # > 12 words
    ]
    return pd.DataFrame([a for a, _ in clips]), pd.DataFrame([i for _, i in clips])


def test_pick_prefers_test_split_clean_5_to_12_words_least_like_bangla() -> None:
    audit, index = _tables()
    picks = baseline.pick_ctg_prompts(audit, index)
    assert list(picks) == ["CTG_1", "CTG_2", "CTG_3"]
    assert sorted(cid for cid, _ in picks.values()) == ["10", "11", "13"]
    sims = [baseline.bangla_similarity(" ".join(text.split()), BN) for _, text in picks.values()]
    assert sims == sorted(sims) and sims[-1] < 1.0  # CTG_1 is the least Bangla-like
    texts = {cid: text for cid, text in picks.values()}
    assert texts["11"] == index.loc[1, "ctg_text"]  # verbatim, incl. the outer spaces
    assert baseline.pick_ctg_prompts(audit, index) == picks  # fixed seed: same picks


def test_pick_falls_back_when_too_few_in_word_range(capsys: pytest.CaptureFixture[str]) -> None:
    audit, index = _tables()
    keep = ["10", "16", "17"]
    picks = baseline.pick_ctg_prompts(audit[audit["clip_id"].isin(keep)],
                                      index[index["clip_id"].isin(keep)])
    assert sorted(cid for cid, _ in picks.values()) == keep
    assert "WARN: only 1 clean test clips with 5-12 words" in capsys.readouterr().out


def test_pick_raises_without_enough_clean_test_clips() -> None:
    audit, index = _tables()
    with pytest.raises(ValueError, match="need 3 clean test-split clips, found 2"):
        baseline.pick_ctg_prompts(audit[audit["clip_id"].isin(["10", "11", "14"])], index)


def test_write_keeps_std_prompts_and_records_sources(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    cfg = paths.configs_dir() / "baseline_prompts.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"STD_1": "আজ", "CTG_1": None, "CTG_2": None, "CTG_3": None},
                              ensure_ascii=False), encoding="utf-8")
    baseline.write_ctg_prompts({"CTG_1": ("10", " আঁই  যাই "), "CTG_2": ("11", "খ"),
                                "CTG_3": ("13", "গ")})
    raw = json.loads(cfg.read_text(encoding="utf-8"))
    assert raw["STD_1"] == "আজ" and raw["CTG_1"] == " আঁই  যাই "
    assert raw["_sources"] == {"CTG_1": "10", "CTG_2": "11", "CTG_3": "13"}
    assert "=== BEGIN configs/baseline_prompts.json ===" in capsys.readouterr().out
    assert baseline.load_prompts() == {"STD_1": "আজ", "CTG_1": " আঁই  যাই ", "CTG_2": "খ",
                                       "CTG_3": "গ"}  # _sources is not a prompt
    assert list(baseline.load_prompts(prefix="CTG_")) == ["CTG_1", "CTG_2", "CTG_3"]


def test_bangla_similarity_and_write_without_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert baseline.bangla_similarity("আমি যাব", " আমি  যাব ") == 1.0  # BN side normalised
    assert baseline.bangla_similarity("আঁই যাইয়ুম", "আমি যাব") < 0.7
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    baseline.write_ctg_prompts({"CTG_1": ("10", "ক")})
    raw = json.loads((paths.configs_dir() / "baseline_prompts.json").read_text(encoding="utf-8"))
    assert raw == {"CTG_1": "ক", "_sources": {"CTG_1": "10"}}


def test_cli_pick_baseline_ctg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    audit, index = _tables()
    paths.drive_reports().mkdir(parents=True)
    audit.to_csv(paths.drive_reports() / data_audit.AUDIT_CSV, index=False)
    monkeypatch.setattr(corpus, "load_index", lambda root=None: index)
    paths.configs_dir().mkdir(parents=True)
    (paths.configs_dir() / "baseline_prompts.json").write_text('{"STD_1": "আজ"}',
                                                                encoding="utf-8")
    assert cli.main(["pick-baseline-ctg"]) == 0
    out = capsys.readouterr().out
    assert "PASS: picked 3 dialect prompts" in out
    assert "CTG_1 clip " in out
    (paths.drive_reports() / data_audit.AUDIT_CSV).unlink()
    assert cli.main(["pick-baseline-ctg"]) == 1
    assert "FAIL: pick-baseline-ctg:" in capsys.readouterr().out
