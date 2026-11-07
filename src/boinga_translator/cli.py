"""Command-line entry point (`boinga`) for data prep, training and demo commands."""

from __future__ import annotations

import argparse

from boinga_translator import __version__, env_report, pretrained


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boinga", description="boinga-translator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the package version")
    sub.add_parser("env", help="record the environment to reports/env.json and print it")
    fetch = sub.add_parser("fetch-pretrained", help="download and cache the pretrained voices")
    fetch.add_argument("--voice", choices=[*pretrained.voices(), "both"], default="both")
    insp = sub.add_parser("inspect-pretrained", help="report config/checkpoint facts per voice")
    insp.add_argument("--voice", choices=[*pretrained.voices(), "both"], default="both")
    return parser


def _inspect_pretrained(voice: str) -> int:
    """Inspect one or both voices, then write the shared reports once."""
    results, status = {}, 0
    for v in pretrained.voices() if voice == "both" else [voice]:
        try:
            results[v] = pretrained.inspect(v)
        except Exception as exc:  # noqa: BLE001 - torch.load/unpickling errors vary; report each
            print(f"FAIL: {v}: {type(exc).__name__}: {exc}")
            status = 1
    if results:
        try:
            pretrained.write_inspect_reports(results)
        except (RuntimeError, OSError) as exc:
            print(f"FAIL: reports: {exc}")
            status = 1
    return status


def _fetch_pretrained(voice: str) -> int:
    """Fetch one or both voices; print FAIL and return 1 if any fails."""
    status = 0
    for v in pretrained.voices() if voice == "both" else [voice]:
        try:
            pretrained.fetch(v)
        except (RuntimeError, ValueError, OSError) as exc:  # report every voice, then fail
            print(f"FAIL: {v}: {exc}")
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(__version__)
    elif args.command == "env":
        env_report.run()
    elif args.command == "fetch-pretrained":
        return _fetch_pretrained(args.voice)
    elif args.command == "inspect-pretrained":
        return _inspect_pretrained(args.voice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
