"""Tests for evaluate.pick_prompts and `boinga pick-eval-prompts`: 15 + 5, no leaks, stable ids."""

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
_TEXTS = (" ".join(p) for p in itertools.permutations(WORDS, 6))  # distinct 6-word lines


def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))


def _tables() -> evaluate.Tables:
    """Own speaker (clips o*): 18 clean test, 1 short, 1 unclean, train/val ones.
    Other speakers (x*): 2 clean test clips in each of clusters 2-7, plus traps."""
    audit: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    texts = iter(_TEXTS)

    def add(cid: str, split_off: str, split: str | None, cluster: int, text: str | None = None,
            bn: str = BN, **kw: Any) -> str:
        text = text if text is not None else next(texts)
        audit.append(_row(cid, split_official=split_off, ctg_text_nfc=" ".join(text.split()),
                          **kw))
        index.append({"clip_id": cid, "ctg_text": text, "bn_text": bn})
        clusters.append({"clip_id": cid, "cluster": cluster})
        if split is not None:
            splits.append({"clip_id": cid, "split": split, "duration_s": 3.0})
        return text

    for i in range(18):
        add(f"o{i}", "test", "test", 1)
    add("o_bn", "test", "test", 1, text=f" {BN} ")  # = its translation: ratio 1.0, ranked last
    add("o_short", "test", "test", 1, text="আঁই যাই")  # 2 words: outside 5-15
    add("o_quiet", "test", "test", 1, rms_db=-60.0)  # not clean
    train_text = add("o_tr", "train", "train", 1)
    add("o_val", "val", "val", 1)
    for k in range(2, 8):
        for j in range(2):
            add(f"x{k}_{j}", "test", None, k)
    add("x_leak", "test", None, 2, text=train_text, bn="অন্য")  # same text as a train clip
    add("x_train", "train", None, 3)  # other speaker, official train split
    add("x_loud", "test", None, 4, clip_ratio=0.01)  # not clean
    add("x_own", "test", None, 1)  # training speaker's voice cluster, outside the selection
    add("o_nobn", "test", "test", 1, bn="")  # no translation to compare with
    own = {r["clip_id"] for r in splits}
    return evaluate.Tables(audit=pd.DataFrame(audit), index=pd.DataFrame(index),
                           splits=pd.DataFrame(splits), own_ids=own,
                           clusters=pd.DataFrame(clusters))


def test_pick_15_own_test_and_5_other_speakers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    picks = evaluate.pick_prompts(tables=tables)
    assert [p["id"] for p in picks] == [f"P{i:02d}" for i in range(1, 21)]
    own, other = picks[:15], picks[15:]
    assert all(p["has_reference"] for p in own) and not any(p["has_reference"] for p in other)
    assert all(p["clip_id"].startswith("o") for p in own)
    assert {"o_quiet", "o_tr", "o_val", "o_short", "o_bn", "o_nobn"}.isdisjoint(
        p["clip_id"] for p in own)
    clusters = tables.clusters.set_index("clip_id")["cluster"]
    assert len({clusters[p["clip_id"]] for p in other}) == 5  # one clip per other speaker
    assert all(p["clip_id"].startswith("x") and p["clip_id"][1] != "_" for p in other)
    for p in picks:  # verbatim text and its translation, nothing else
        row = tables.index.set_index("clip_id").loc[p["clip_id"]]
        assert (p["ctg_text"], p["bn_text"]) == (row["ctg_text"], row["bn_text"])
        assert set(p) == {"id", "clip_id", "ctg_text", "bn_text", "has_reference"}
    sims = [evaluate.bangla_similarity(" ".join(p["ctg_text"].split()), BN) for p in own]
    assert sims == sorted(sims)  # most dialectal first
    assert evaluate.pick_prompts(tables=tables) == picks  # fixed seed: same picks


def test_no_overlap_with_train_or_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    picks = evaluate.pick_prompts(tables=tables)
    split = tables.splits.set_index("clip_id")["split"]
    seen = tables.audit.merge(tables.splits[tables.splits["split"] != "test"], on="clip_id")
    assert all(split.get(p["clip_id"], "none") in ("test", "none") for p in picks)
    assert {" ".join(p["ctg_text"].split()) for p in picks}.isdisjoint(seen["ctg_text_nfc"])
    assert len({p["clip_id"] for p in picks}) == 20


def test_short_lines_only_when_too_few_in_range(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    drop = {f"o{i}" for i in range(5)}  # 13 in-range + o_bn + o_short left: all 15 needed
    tables = evaluate.Tables(audit=tables.audit[~tables.audit["clip_id"].isin(drop)],
                             index=tables.index, splits=tables.splits, own_ids=tables.own_ids,
                             clusters=tables.clusters)
    own = [p["clip_id"] for p in evaluate.pick_prompts(tables=tables)[:15]]
    assert own[-1] == "o_short" and own[-2] == "o_bn"  # in-range first, then by ratio


def test_exclude_replaces_only_those_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    first = evaluate.pick_prompts(tables=tables)
    evaluate.write_prompts(first)
    assert "=== BEGIN configs/eval_prompts.json ===" in capsys.readouterr().out
    second = evaluate.pick_prompts(exclude_ids=["P03", "P17"], tables=tables)
    changed = [a["id"] for a, b in zip(first, second, strict=True) if a != b]
    assert changed == ["P03", "P17"]
    used = {p["clip_id"] for p in first}
    assert second[2]["clip_id"] not in used and second[2]["has_reference"]
    assert second[16]["clip_id"] not in used and not second[16]["has_reference"]
    evaluate.write_prompts(second)
    third = evaluate.pick_prompts(exclude_ids=["P11"], tables=tables)  # no excluded clip returns
    assert third[10]["clip_id"] not in used | {p["clip_id"] for p in second}
    assert evaluate.pick_prompts(tables=tables) == second  # a plain re-run keeps the file


def test_other_speaker_replacement_keeps_one_per_cluster(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    five = evaluate.Tables(audit=tables.audit[~tables.audit["clip_id"].str.startswith("x7")],
                           index=tables.index, splits=tables.splits, own_ids=tables.own_ids,
                           clusters=tables.clusters)  # exactly 5 other clusters
    first = evaluate.pick_prompts(tables=five)
    evaluate.write_prompts(first)
    second = evaluate.pick_prompts(exclude_ids=["P17"], tables=five)
    cluster = five.clusters.set_index("clip_id")["cluster"]
    assert cluster[second[16]["clip_id"]] == cluster[first[16]["clip_id"]]
    assert len({cluster[p["clip_id"]] for p in second[15:]}) == 5


def test_exclude_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    with pytest.raises(ValueError, match="no configs/eval_prompts.json"):
        evaluate.pick_prompts(exclude_ids=["P01"], tables=tables)
    evaluate.write_prompts(evaluate.pick_prompts(tables=tables))
    with pytest.raises(ValueError, match="unknown prompt id"):
        evaluate.pick_prompts(exclude_ids=["P21"], tables=tables)
    # own speaker: 20 candidates (18 + o_bn + o_short), 15 used, 5 spare
    evaluate.write_prompts(evaluate.pick_prompts(exclude_ids=["P01", "P02", "P03", "P04", "P05"],
                                                 tables=tables))
    with pytest.raises(ValueError, match="no candidate left"):
        evaluate.pick_prompts(exclude_ids=["P06"], tables=tables)


def test_too_few_candidates_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, monkeypatch)
    tables = _tables()
    small = evaluate.Tables(audit=tables.audit[~tables.audit["clip_id"].str.startswith("x")],
                            index=tables.index, splits=tables.splits, own_ids=tables.own_ids,
                            clusters=tables.clusters)
    with pytest.raises(ValueError, match="need 5 other-speaker"):
        evaluate.pick_prompts(tables=small)


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
    assert shown.count("\n") >= 20 and second[0]["ctg_text"] in shown
