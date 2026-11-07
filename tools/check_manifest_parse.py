"""Colab check (C6.3): coqui's ljspeech formatter parses our train/val/test manifests."""

from __future__ import annotations

import sys

from boinga_translator import prepare


def main() -> int:
    """Run prepare.parse_check() on paths.data_prepared(); print FAIL and return 1 on error."""
    try:
        prepare.parse_check()
    # coqui asserts a non-empty manifest and indexes cols[2] (IndexError on a short line)
    except (OSError, ValueError, RuntimeError, ImportError, AssertionError, IndexError) as exc:
        print(f"FAIL: manifest parse check: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
