"""Command-line entry point (`boinga`) for data prep, training and demo commands."""

from __future__ import annotations

import argparse

from boinga_translator import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boinga", description="boinga-translator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the package version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
