"""Tests for text_frontend.py: NFC-only normalise(), char_map application, row exclusion."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from boinga_translator import data_audit, text_frontend

# --- normalise(): NFC + collapse whitespace + strip, nothing else -------------------------
HASANTA, NUKTA, KHANDA_TA = "্", "়", "ৎ"
ZWJ, ZWNJ = "‍", "‌"
DIALECT = [  # already NFC and single-spaced: normalise() must return them unchanged
    "আঁই বাজারত",  # chandrabindu, vowel signs
    f"ক{HASANTA}ষ",  # hasanta conjunct
    f"ড{NUKTA}া ঢ{NUKTA} য{NUKTA}",  # base + nukta: NFC of U+09DC/U+09DD/U+09DF
    f"উ{KHANDA_TA}সব ত{HASANTA}{ZWJ} ক{HASANTA}{ZWNJ}ষ",
    "কোন কৌশল ঋণ দু’রাত।",
]


def test_dialect_fixture_is_nfc() -> None:
    for text in DIALECT:
        assert unicodedata.is_normalized("NFC", text), text


@pytest.mark.parametrize("text", DIALECT)
def test_normalise_keeps_nfc_dialect_text_unchanged(text: str) -> None:
    assert text_frontend.normalise(text) == text


@pytest.mark.parametrize("c", [HASANTA, NUKTA, KHANDA_TA, ZWJ, ZWNJ])
def test_normalise_preserves_marks_and_joiners(c: str) -> None:
    assert c in text_frontend.normalise(f"ক{c}ষ")


@pytest.mark.parametrize("text", [*DIALECT, "  আঁই \t ব\n\nয ম ",
                                  "ড়ঢ়য়", "কো"])
def test_normalise_is_idempotent(text: str) -> None:
    once = text_frontend.normalise(text)
    assert text_frontend.normalise(once) == once


def test_normalise_collapses_and_strips_whitespace_only() -> None:
    assert text_frontend.normalise("  a \t b\n\nc  ") == "a b c"
    assert text_frontend.normalise("আ  ই ।") == "আ ই ।"  # NBSP
    assert text_frontend.normalise("।'(–)\"|") == "।'(–)\"|"  # char_map's job
    assert text_frontend.normalise("") == ""


def test_python_nfc_decomposes_precomposed_nukta_letters() -> None:
    # U+09DC/U+09DD/U+09DF are Unicode composition exclusions: NFC turns the precomposed letter
    # into base + nukta and never recomposes it, so both spellings meet in the decomposed form.
    # configs/char_map.json maps them back to precomposed for the model (DECISIONS
    # "Character map"), which is why prepare() output is not NFC.
    assert text_frontend.normalise("ড়") == "ড়"  # RRA -> DDA + NUKTA
    assert text_frontend.normalise("ড়") == "ড়"  # unchanged
    assert text_frontend.normalise("ঢ়") == "ঢ়"
    assert text_frontend.normalise("য়") == "য়"
    # NFC also joins a two-part vowel sign: e-kar + aa-kar -> o-kar (U+09CB).
    assert text_frontend.normalise("কো") == "কো"

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
