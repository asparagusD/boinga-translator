"""Tests for text_frontend.py: char_map application, row exclusion and idempotence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boinga_translator import data_audit, text_frontend

YA_PRE, YA_DEC = "য়", "য়"  # precomposed vs NFC (decomposed) য়


def test_real_map_applies_decisions() -> None:
    assert text_frontend.prepare(f"আঁই যাই{YA_DEC}ুম।") == f"আঁই যাই{YA_PRE}ুম."
    assert text_frontend.prepare("ভালা আছোন৷") == "ভালা আছোন."
    assert text_frontend.prepare("আগ্ৰহ") == "আগ্রহ"
    assert text_frontend.prepare("কমিশন (ইসি) সংলাপ") == "কমিশন ইসি সংলাপ"
    assert text_frontend.prepare(" 'চাটগাঁ' বই ") == "চাটগাঁ বই"  # stripped + collapsed
    assert text_frontend.prepare("হূনুন…।") == "হূনুন."
    assert text_frontend.prepare("লেহা থাহে–ইবা") == "লেহা থাহে—ইবা"
    assert text_frontend.prepare("দু’রাত") == "দু’রাত"  # kept on purpose


@pytest.mark.parametrize("text", [f"যা{YA_DEC}", f"যা{YA_PRE}"])
def test_nukta_letters_end_up_precomposed(text: str) -> None:
    assert text_frontend.prepare(text) == f"যা{YA_PRE}"
    assert text_frontend.prepare("পড়া") == "পড়া"


SAMPLES = ["আঁই যাইয়ুম।", f"যা{YA_PRE} \"কতা\" (আর)  যা{YA_DEC} – ঘর ।", "দু’রাত | যাই; আর",
           "অ‍ন্য", " ভালা৷ "]


@pytest.mark.parametrize("text", SAMPLES)
def test_prepare_is_idempotent_and_known_to_both_voices(text: str) -> None:
    once = text_frontend.prepare(text)
    assert text_frontend.prepare(once) == once
    for voice in ("female", "male"):
        known = data_audit.load_model_charset(voice)
        assert {c for c in once if c != " "} <= known, (voice, once)


def test_excluded_rows() -> None:
    for text in ("দু:খ আছে।", "দিন/সপ্তার", "২০১৯ সালর", "7টা", "গু* লিতে"):
        assert text_frontend.excluded(text), text
    assert not text_frontend.excluded("আঁই যাইয়ুম।")


def test_custom_map_file_and_validation(tmp_path: Path) -> None:
    good = tmp_path / "m.json"
    good.write_text(json.dumps({"map": [{"from": "a", "to": "b", "note": "x"}],
                                "exclude_rows_with": {"chars": "#", "note": "y"}}),
                    encoding="utf-8")
    cmap = text_frontend.load_char_map(good)
    assert text_frontend.prepare("  a  a ", cmap) == "b b"
    assert text_frontend.excluded("x#", cmap)
    assert text_frontend.map_digest(cmap) != text_frontend.map_digest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"map": [{"from": "", "to": "b", "note": "x"}],
                               "exclude_rows_with": {"chars": "", "note": ""}}), encoding="utf-8")
    with pytest.raises(ValueError, match="empty 'from'"):
        text_frontend.load_char_map(bad)
