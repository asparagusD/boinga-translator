"""Tests for configs/char_map.json: entry shape, targets inside both voices' charsets, exclusions."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

import pytest

from boinga_translator import data_audit, paths


@pytest.fixture(scope="module")
def char_map() -> dict[str, Any]:
    return json.loads((paths.configs_dir() / "char_map.json").read_text(encoding="utf-8"))


def test_entries_are_complete_and_unique(char_map: dict[str, Any]) -> None:
    entries = char_map["map"]
    assert entries, "char_map has no entries"
    for e in entries:
        assert set(e) == {"from", "to", "note"}
        assert isinstance(e["from"], str) and e["from"]
        assert isinstance(e["to"], str)
        assert e["note"].strip()  # every mapping says why (dialect-data-rules)
    froms = [e["from"] for e in entries]
    assert len(froms) == len(set(froms))


def test_sources_can_match_nfc_text(char_map: dict[str, Any]) -> None:
    """The map runs on NFC text, so a key that NFC would rewrite could never match."""
    for e in char_map["map"]:
        assert unicodedata.normalize("NFC", e["from"]) == e["from"], e


def test_no_output_feeds_a_later_entry(char_map: dict[str, Any]) -> None:
    """Order guard: a replacement never creates the source of a later mapping."""
    entries = char_map["map"]
    for i, e in enumerate(entries):
        for later in entries[i + 1:]:
            assert later["from"] not in e["to"], (e, later)


@pytest.mark.parametrize("voice", ["female", "male"])
def test_targets_use_only_known_characters(char_map: dict[str, Any], voice: str) -> None:
    known = data_audit.load_model_charset(voice)
    for e in char_map["map"]:
        assert set(e["to"]) <= known, (voice, e)


def test_exclusions(char_map: dict[str, Any]) -> None:
    excl = char_map["exclude_rows_with"]
    assert excl["note"].strip()
    chars = set(excl["chars"])
    assert {"/", ":", "*", "0", "9", "০", "৯"} <= chars
    assert not any(set(e["to"]) & chars for e in char_map["map"])  # never map into an exclusion
    assert not any(e["from"] in chars for e in char_map["map"])  # a char is mapped or excluded


def test_pinned_decisions(char_map: dict[str, Any]) -> None:
    to = {e["from"]: e["to"] for e in char_map["map"]}
    assert to["।"] == "." and to["৷"] == "."  # danda and its lookalike
    assert to["ৰ"] == "র" and to["ৱ"] == "র"  # Assamese RA/WA typos
    nukta = {"য়": "য়", "ড়": "ড়", "ঢ়": "ঢ়"}
    for src, dst in nukta.items():
        assert to[src] == dst
        assert unicodedata.normalize("NFC", dst) == src  # really precomposed: NFC undoes it
    assert "’" not in to and "‌" not in to  # kept on purpose (both voices know them)
