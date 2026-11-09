"""Tests for README.md: relative links resolve, every `boinga` subcommand is listed and real."""

from __future__ import annotations

import re

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
