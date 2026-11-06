"""Commit-msg guard: enforce the atomic-commit message format (header, Chunk/What/Why/How verified)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TYPES = "feat|fix|refactor|test|docs|chore|perf|data|train|eval"
HEADER_RE = re.compile(rf"^({TYPES})\([a-z0-9-]+\): (.{{1,72}})$")
MERGE_RE = re.compile(r"^merge\((C\d+|F)\): .+")
CHUNK_RE = re.compile(r"^Chunk: (C\d+\.\d+[a-z]?|F\.\d+)$")
SECTIONS = ("What:", "Why:", "How verified:")
BANNED = {"wip", "update", "fix", "changes", "misc", "stuff"}

TEMPLATE = """\
<type>(<scope>): <imperative summary, <= 72 chars, no trailing period>

Chunk: C<n>.<m>[<letter>]
What:
- <concrete change 1, name files/functions>
Why:
- <reason, link to plan section or decision>
How verified:
- <exact command> -> <real result, e.g. "pytest -q -> 14 passed in 2.1s">
Notes:
- <limitations, follow-ups, or "none">

type: feat, fix, refactor, test, docs, chore, perf, data, train, eval"""


def validate(message: str) -> list[str]:
    """Return a list of reasons the message is invalid (empty list = valid)."""
    lines = [ln.rstrip() for ln in message.splitlines() if not ln.startswith("#")]
    while lines and not lines[0]:
        lines.pop(0)
    if not lines:
        return ["empty commit message"]
    header, body = lines[0], lines[1:]
    if MERGE_RE.match(header):
        ok = any(ln.startswith("- ") for ln in body)
        return [] if ok else ["merge commit body needs at least one '- ' bullet"]

    errors = []
    m = HEADER_RE.match(header)
    summary = m.group(2) if m else header
    if summary.strip().rstrip(".").lower() in BANNED:
        errors.append(f"summary '{summary.strip()}' is too vague; say what changed")
    if not m:
        errors.append("header must be '<type>(<scope>): <summary>' with a known type, "
                      "lowercase scope and a summary of at most 72 chars")
    if header.endswith("."):
        errors.append("header must not end with a period")
    if not any(CHUNK_RE.match(ln) for ln in body):
        errors.append("missing 'Chunk: C<n>.<m>[a-z]' (or 'Chunk: F.<n>') line")
    for section in SECTIONS:
        if section not in body:
            errors.append(f"missing '{section}' section")
            continue
        i = body.index(section)
        if i + 1 >= len(body) or not body[i + 1].startswith("- "):
            errors.append(f"'{section}' must be followed by at least one '- ' bullet")
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    message = Path(argv[0]).read_text(encoding="utf-8")
    errors = validate(message)
    if not errors:
        return 0
    print("Commit message rejected:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    print("\nUse this format (atomic-commit skill):\n\n" + TEMPLATE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
