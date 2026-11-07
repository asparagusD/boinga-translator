"""Command-line entry point (`boinga`) for data prep, training and demo commands."""

from __future__ import annotations

import argparse

from boinga_translator import (
    __version__,
    audit_summary,
    baseline,
    corpus,
    data_audit,
    data_download,
    env_report,
    paths,
    pretrained,
    speaker_select,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boinga", description="boinga-translator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the package version")
    sub.add_parser("env", help="record the environment to reports/env.json and print it")
    fetch = sub.add_parser("fetch-pretrained", help="download and cache the pretrained voices")
    fetch.add_argument("--voice", choices=[*pretrained.voices(), "both"], default="both")
    insp = sub.add_parser("inspect-pretrained", help="report config/checkpoint facts per voice")
    insp.add_argument("--voice", choices=[*pretrained.voices(), "both"], default="both")
    base = sub.add_parser("baseline", help="synthesise the baseline prompts with each voice")
    base.add_argument("--voice", choices=[*pretrained.voices(), "both"], default="both")
    sub.add_parser("fetch-data", help="download, verify and unpack the corpus (resumable)")
    sub.add_parser("index", help="index the corpus: wav, dialect text, translation, split")
    aud = sub.add_parser("audit", help="audit every clip to Drive reports/audit.csv (resumable)")
    # Gate G1 recorded no preferred voice (quality "same"), so the default is female.
    aud.add_argument("--voice", choices=pretrained.voices(), default="female")
    sub.add_parser("audit-summary", help="summarise audit.csv into reports/audit_summary.md")
    sub.add_parser("pick-baseline-ctg",
                   help="put 3 held-out dialect lines into configs/baseline_prompts.json")
    sub.add_parser("embed-speakers",
                   help="embed every clean clip's speaker to Drive speaker_emb/ (resumable)")
    return parser


def _pick_baseline_ctg() -> int:
    """Pick CTG_1..3 from clean test clips and write them; FAIL + exit 1 on error."""
    try:
        picks = baseline.pick_ctg_prompts(
            data_audit.read_audit(paths.drive_reports() / data_audit.AUDIT_CSV),
            corpus.load_index())
        for key, (clip_id, text) in picks.items():
            print(f"{key} clip {clip_id}: {text}")
        baseline.write_ctg_prompts(picks)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: pick-baseline-ctg: {type(exc).__name__}: {exc}")
        return 1
    print(f"PASS: picked {len(picks)} dialect prompts")
    return 0


def _embed_speakers() -> int:
    """One speaker embedding per clean clip of the Drive audit.csv; FAIL + exit 1 on error."""
    try:
        speaker_select.embed(data_audit.read_audit(paths.drive_reports() / data_audit.AUDIT_CSV))
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
        print(f"FAIL: embed-speakers: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _audit_summary() -> int:
    """Summarise the Drive audit.csv; FAIL + exit 1 if it is missing or unreadable."""
    try:
        audit_summary.summarise()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: audit-summary: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _audit(voice: str) -> int:
    """Index then audit every clip against the voice's charset; FAIL + exit 1 on error."""
    print(f"voice {voice}")
    try:
        data_audit.audit(corpus.load_index(), voice)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"FAIL: audit: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _index() -> int:
    """Build the corpus index and print its summary; FAIL + exit 1 if the layout is off."""
    try:
        df = corpus.load_index()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"FAIL: index: {exc}")
        return 1
    print("\n".join(corpus.describe(df)))
    print(f"PASS: index has {len(df)} rows ({int(df['wav_exists'].sum())} with audio)")
    return 0


def _fetch_data() -> int:
    """Fetch the corpus; print FAIL and return 1 on a bad download or I/O error."""
    try:
        data_download.fetch()
    except (RuntimeError, OSError) as exc:  # urllib's URLError is an OSError
        print(f"FAIL: fetch-data: {exc}")
        return 1
    return 0


def _baseline(voice: str) -> int:
    """Run the baseline per voice; exit 1 if any voice fails or errors."""
    status = 0
    for v in pretrained.voices() if voice == "both" else [voice]:
        try:
            ok = baseline.run(v)
        except Exception as exc:  # noqa: BLE001 - coqui/torch errors vary; report each voice
            print(f"FAIL: baseline {v}: {type(exc).__name__}: {exc}")
            ok = False
        status = status or (0 if ok else 1)
    return status


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
    elif args.command == "baseline":
        return _baseline(args.voice)
    elif args.command == "fetch-data":
        return _fetch_data()
    elif args.command == "index":
        return _index()
    elif args.command == "audit":
        return _audit(args.voice)
    elif args.command == "audit-summary":
        return _audit_summary()
    elif args.command == "pick-baseline-ctg":
        return _pick_baseline_ctg()
    elif args.command == "embed-speakers":
        return _embed_speakers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
