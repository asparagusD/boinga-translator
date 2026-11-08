"""Tests for evaluate.pick_prompts and `boinga pick-eval-prompts`: 20 test clips, no leaks, ids."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from boinga_translator import cli, evaluate, paths
from test_audit_summary import _row  # one audit row with sane defaults

BN = "আমি আজ সকালে বাজারে গিয়ে তাজা সবজি কিনেছি"
WORDS = ["আঁই", "হেতে", "বাজারত", "তরকারি", "কিনগি", "আজিয়া", "বিয়ানে", "ঘরত"]
KEYS = {"id", "clip_id", "ctg_text", "bn_text", "has_reference"}


def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))


def _tables(n_good: int = 23) -> evaluate.Tables:
    """n_good clean 6-word test clips (t*) plus clips that must never be picked."""
    audit: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    texts = (" ".join(p) for p in itertools.permutations(WORDS, 6))  # distinct 6-word lines

    def add(cid: str, split: str | None, text: str | None = None, bn: str = BN,
            **kw: Any) -> str:
        text = text if text is not None else next(texts)
        audit.append(_row(cid, split_official=split or "test",
                          ctg_text_nfc=" ".join(text.split()), **kw))
        index.append({"clip_id": cid, "ctg_text": text, "bn_text": bn})
        if split is not None:
            splits.append({"clip_id": cid, "split": split, "duration_s": 3.0})
        return text

    for i in range(n_good):
        add(f"t{i}", "test")
    add("t_bn", "test", text=f" {BN} ")  # = its translation: ratio 1.0, ranked last in range
    add("t_short", "test", text="আঁই যাই")  # 2 words: outside 5-15
    add("t_quiet", "test", rms_db=-60.0)  # not clean
    add("t_nobn", "test", bn="")  # no translation to compare with
    train_text = add("tr", "train")
    add("va", "val")
    add("t_leak", "test", text=train_text)  # same text as a train clip
    add("none", None)  # not in our splits (other speaker)
    add("t_dup", "test", text=index[0]["ctg_text"])  # same text as t0: only one is a candidate
    return evaluate.Tables(audit=pd.DataFrame(audit), index=pd.DataFrame(index),
                           splits=pd.DataFrame(splits))


def test_pick_20_test_split_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    picks = evaluate.pick_prompts(tables=tables)
    assert [p["id"] for p in picks] == [f"P{i:02d}" for i in range(1, 21)]
    assert all(p["has_reference"] and set(p) == KEYS for p in picks)
    assert all(p["clip_id"].startswith("t") and p["clip_id"][1] != "_" for p in picks)
    assert "t_dup" not in evaluate.candidates(tables)  # distinct text
    for p in picks:  # verbatim text and its translation, nothing else
        row = tables.index.set_index("clip_id").loc[p["clip_id"]]
        assert (p["ctg_text"], p["bn_text"]) == (row["ctg_text"], row["bn_text"])
    sims = [evaluate.bangla_similarity(" ".join(p["ctg_text"].split()), BN) for p in picks]
    assert sims == sorted(sims)  # most dialectal first
    assert evaluate.pick_prompts(tables=tables) == picks  # fixed seed: same picks


def test_no_overlap_with_train_or_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    picks = evaluate.pick_prompts(tables=tables)
    split = tables.splits.set_index("clip_id")["split"]
    seen = tables.audit.merge(tables.splits[tables.splits["split"] != "test"], on="clip_id")
    assert all(split[p["clip_id"]] == "test" for p in picks)
    assert {" ".join(p["ctg_text"].split()) for p in picks}.isdisjoint(seen["ctg_text_nfc"])
    assert len({p["clip_id"] for p in picks}) == 20


def test_short_lines_only_when_too_few_in_range(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    ids = [p["clip_id"] for p in evaluate.pick_prompts(tables=_tables(n_good=18))]
    assert ids[-2:] == ["t_bn", "t_short"]  # in-range first, then by ratio


def test_exclude_replaces_only_those_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    first = evaluate.pick_prompts(tables=tables)
    evaluate.write_prompts(first)
    assert "=== BEGIN configs/eval_prompts.json ===" in capsys.readouterr().out
    second = evaluate.pick_prompts(exclude_ids=["P03", "P17"], tables=tables)
    assert [a["id"] for a, b in zip(first, second, strict=True) if a != b] == ["P03", "P17"]
    used = {p["clip_id"] for p in first}
    assert {second[2]["clip_id"], second[16]["clip_id"]}.isdisjoint(used)
    assert second[2]["replaced"] == [first[2]["clip_id"]]
    evaluate.write_prompts(second)
    third = evaluate.pick_prompts(exclude_ids=["P11"], tables=tables)  # no excluded clip returns
    assert third[10]["clip_id"] not in used | {p["clip_id"] for p in second}
    assert evaluate.pick_prompts(tables=tables) == second  # a plain re-run keeps the file


def test_exclude_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()  # 25 candidates: 23 + t_bn + t_short; 20 used, 5 spare
    with pytest.raises(ValueError, match="no configs/eval_prompts.json"):
        evaluate.pick_prompts(exclude_ids=["P01"], tables=tables)
    evaluate.write_prompts(evaluate.pick_prompts(tables=tables))
    with pytest.raises(ValueError, match="unknown prompt id"):
        evaluate.pick_prompts(exclude_ids=["P21"], tables=tables)
    evaluate.write_prompts(evaluate.pick_prompts(exclude_ids=["P01", "P02", "P03", "P04", "P05"],
                                                 tables=tables))
    with pytest.raises(ValueError, match="no candidate left"):
        evaluate.pick_prompts(exclude_ids=["P06"], tables=tables)


def test_stale_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    picks = evaluate.pick_prompts(tables=_tables())
    evaluate.write_prompts([{**picks[0], "clip_id": "t_quiet"}, *picks[1:]])
    with pytest.raises(ValueError, match="P01 in configs/eval_prompts.json are no longer"):
        evaluate.pick_prompts(tables=_tables())


def test_too_few_candidates_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="need 20 test-split candidates, found 19"):
        evaluate.pick_prompts(tables=_tables(n_good=17))


def test_cli_pick_eval_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    monkeypatch.setattr(evaluate, "load_tables", lambda: tables)
    assert cli.main(["pick-eval-prompts"]) == 0
    out = capsys.readouterr().out
    assert "PASS: 20 prompts" in out and "P01 " in out
    first = json.loads((paths.configs_dir() / "eval_prompts.json").read_text(encoding="utf-8"))
    assert cli.main(["pick-eval-prompts", "--exclude", "P03,P11"]) == 0
    second = json.loads((paths.configs_dir() / "eval_prompts.json").read_text(encoding="utf-8"))
    assert [a["id"] for a, b in zip(first, second, strict=True) if a != b] == ["P03", "P11"]
    assert cli.main(["pick-eval-prompts", "--exclude", "P99"]) == 1
    assert "FAIL: pick-eval-prompts:" in capsys.readouterr().out
    evaluate.show_prompts()
    shown = capsys.readouterr().out
    assert shown.count("\n") >= 60 and second[0]["ctg_text"] in shown
