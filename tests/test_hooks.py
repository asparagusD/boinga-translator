"""Tests for tools/hooks/commit_msg.py: valid messages pass, bad ones fail with the right reason."""

from __future__ import annotations

from pathlib import Path

import pytest
from hooks import commit_msg

GOOD = """\
feat(data-audit): compute per-clip duration, rms and clipping ratio

Chunk: C1.2a
What:
- src/boinga_translator/audit.py: clip_stats()
Why:
- Needed to filter clipped audio
How verified:
- pytest -q -> 14 passed in 2.1s
Notes:
- none
"""


def test_good_message_passes() -> None:
    assert commit_msg.validate(GOOD) == []


def test_fallback_chunk_and_merge_commit_pass() -> None:
    assert commit_msg.validate(GOOD.replace("Chunk: C1.2a", "Chunk: F.3")) == []
    assert commit_msg.validate("merge(C0): working system\n\n- C0.1 skeleton\n") == []


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("update\n", "too vague"),
        (GOOD.replace("feat(data-audit)", "feature(Data)"), "header must be"),
        (GOOD.replace("clipping ratio\n", "clipping ratio.\n", 1), "must not end with a period"),
        (GOOD.replace("Chunk: C1.2a\n", ""), "missing 'Chunk:"),
        (GOOD.replace("What:\n- src/boinga_translator/audit.py: clip_stats()\n", "What:\n"),
         "'What:' must be followed by at least one '- ' bullet"),
    ],
)
def test_bad_messages_fail_with_reason(message: str, reason: str) -> None:
    errors = commit_msg.validate(message)
    assert any(reason in e for e in errors), errors


def test_main_rejects_and_prints_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text("wip\n", encoding="utf-8")
    assert commit_msg.main([str(msg)]) == 1
    err = capsys.readouterr().err
    assert "too vague" in err and "How verified:" in err
