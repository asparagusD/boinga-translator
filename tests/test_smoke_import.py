"""Smoke test: the package imports and exposes a version."""

import boinga_translator


def test_version() -> None:
    assert boinga_translator.__version__ == "0.1.0"
