"""Tests for README.md: links, `boinga` subcommands, result numbers traced to reports, licences."""

from __future__ import annotations

import json
import re
from typing import Any

from boinga_translator import cli, paths

README = paths.REPO / "README.md"
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_CMD = re.compile(r"(?:^|`|!)boinga ([a-z][a-z-]*)", re.MULTILINE)  # code spans and lines


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _subcommands() -> set[str]:
    parser = cli.build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    return set(action.choices)


def test_relative_links_resolve() -> None:
    missing = []
    for target in _LINK.findall(_text()):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#", 1)[0]
        if path and not (paths.REPO / path).exists():
            missing.append(target)
    assert missing == []


def test_every_named_command_exists() -> None:
    named = set(_CMD.findall(_text()))
    assert named and named - _subcommands() == set()


def test_every_subcommand_is_documented() -> None:
    text = _text()
    undocumented = {c for c in _subcommands()
                    if not re.search(rf"`boinga {re.escape(c)}(?=[ `])", text)}
    assert undocumented == set()


def test_developer_setup_kept() -> None:
    assert "## Developer setup" in _text()


# --- C13.2b: results, licences and citation stay traced to their source files ---

def _flat() -> str:
    """README with every whitespace run as one space, so wrapped phrases still match."""
    return " ".join(_text().split())


def _json(rel: str) -> dict[str, Any]:
    return json.loads((paths.REPO / rel).read_text(encoding="utf-8"))


def test_eval_rows_copied_verbatim() -> None:
    eval_md = (paths.REPO / "reports" / "eval.md").read_text(encoding="utf-8")
    rows = [line for line in eval_md.splitlines()
            if re.match(r"\| (intelligibility|naturalness|dialect_\w+|overall) \|", line)]
    assert len(rows) == 5
    text = _text()
    assert [r for r in rows if r not in text] == []
    win = re.search(r"win rate over prompts: (\d\.\d+)", eval_md)
    assert win and f"win rate {win.group(1)}" in text


def test_training_numbers_match_reports() -> None:
    text = _flat()
    splits = _json("configs/splits_summary.json")
    for split in ("train", "val", "test"):
        assert f"{splits['counts'][split]:,} clips, {splits['minutes'][split]} min" in text
    prov = _json("reports/best_provenance.json")
    thr = _json("reports/throughput.json")
    assert f"{prov['step']:,} steps" in text and f"loss_mel {prov['eval']['loss_mel']}" in text
    assert thr["gpu"] in text and f"{thr['sec_per_step_mean']} s/step" in text
    sel = _json("configs/speaker_selection.json")
    assert f"{sel['n_clips']:,} clips, {sel['clean_minutes']} clean minutes" in text
    assert f"batch {thr['batch_size']}" in text and f"{thr['peak_mem_gb']} GB" in text
    assert f"(at step {prov['eval']['step']:,})" in text
    eval_md = (paths.REPO / "reports" / "eval.md").read_text(encoding="utf-8")
    a_mcd, b_mcd = re.search(r"\| mean ± sd \| (\S+ ± \S+) \| (\S+ ± \S+) \|", eval_md).groups()
    assert f"{b_mcd} for B vs {a_mcd} for A" in text


def test_licences_and_attribution() -> None:
    from boinga_translator import settings

    text = _flat()
    for needle in (settings.DATASET_DOI, settings.DATASET_NAME, settings.PRETRAINED_AUTHOR,
                   "10.17632/2fv6vf9v2z.4", "CC BY 4.0", "Apache-2.0", "MPL-2.0",
                   *settings.DATASET_AUTHORS, *settings.PRETRAINED_IDS.values()):
        assert needle in text, needle
    assert "@software{" in text and "@misc{" in text
    licence = (paths.REPO / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in licence and "Version 2.0, January 2004" in licence
