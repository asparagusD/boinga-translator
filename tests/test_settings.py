"""Tests for settings.py: pinned hash format, https URLs, both voices, seed and the coqui pin."""

from __future__ import annotations

import re

from boinga_translator import paths, settings


def test_dataset_sha256_is_64_lowercase_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", settings.DATASET_SHA256)


def test_urls_are_https() -> None:
    urls = [settings.DATASET_ZIP_URL, *settings.PRETRAINED_RELEASE_URLS.values()]
    assert all(u.startswith("https://") for u in urls)


def test_both_voices_in_both_dicts() -> None:
    assert set(settings.PRETRAINED_IDS) == {"female", "male"}
    assert set(settings.PRETRAINED_RELEASE_URLS) == {"female", "male"}


def test_seed_is_int() -> None:
    assert isinstance(settings.SEED, int) and not isinstance(settings.SEED, bool)


def test_coqui_version_matches_pyproject_train_pin() -> None:
    text = (paths.REPO / "pyproject.toml").read_text(encoding="utf-8")
    pin = re.escape(f'"coqui-tts=={settings.COQUI_TTS_VERSION}"')
    assert re.search(rf"^train = \[[^\]]*{pin}[^\]]*\]", text, re.MULTILINE)
