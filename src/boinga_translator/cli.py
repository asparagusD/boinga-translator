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
    prepare,
    pretrained,
    speaker_select,
    text_frontend,
    text_oov,
    train_config,
    train_runner,
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
    sub.add_parser("cluster-speakers",
                   help="cluster the embeddings by voice into reports/speaker_ranking.md")
    save = sub.add_parser("save-speaker",
                          help="save a cluster as the training speaker (configs/speaker_selection.json)")
    save.add_argument("--id", type=int, required=True, help="cluster rank from speaker_ranking.md")
    save.add_argument("--base-voice", choices=pretrained.voices(), required=True,
                      help="pretrained voice to fine-tune: the speaker's gender")
    sub.add_parser("oov", help="report characters of the selected speaker's text unknown "
                               "to the base voice (reports/oov_report.md)")
    sub.add_parser("check-tokenizer", help="round-trip the selected speaker's text through "
                                           "the base voice's coqui tokenizer")
    prep = sub.add_parser("prepare", help="splits, model-rate wavs, ljspeech manifests and a "
                                          "Drive backup (resumable, restores on a new VM)")
    prep.add_argument("--workers", type=int, default=2, help="transform processes (default 2)")
    bcfg = sub.add_parser("build-config", help="derive configs/vits_ctg_finetune.json from the "
                                               "base voice's pretrained config")
    bcfg.add_argument("--run", required=True, help="run name (one folder under runs/)")
    sub.add_parser("check-loader", help="load one coqui train batch from the fine-tune config "
                                        "and check token ids and finite audio")
    train = sub.add_parser("train", help="start a new fine-tune run from the pretrained "
                                         "checkpoint (run folder on Drive)")
    train.add_argument("--run", required=True, help="run name (one folder under runs/)")
    train.add_argument("--max-steps", type=int, default=None,
                       help="stop after N optimiser steps and save a checkpoint")
    return parser


def _prepare(workers: int) -> int:
    """Build the prepared dataset; FAIL + exit 1 on error."""
    try:
        prepare.build(workers=workers)
    except (ImportError, RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: prepare: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _build_config(run_name: str) -> int:
    """Write the fine-tune config; FAIL + exit 1 on error."""
    try:
        train_config.build(run_name)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:  # JSONDecodeError is a ValueError
        print(f"FAIL: build-config: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _check_loader() -> int:
    """One-batch loader check; FAIL + exit 1 on a bad batch or any error."""
    try:
        return 0 if train_runner.check_loader() else 1
    except Exception as exc:  # noqa: BLE001 - coqui/torch errors vary; show the type
        print(f"FAIL: loader {type(exc).__name__}: {exc}")
        return 1


def _train(run_name: str, max_steps: int | None) -> int:
    """Start a fine-tune run; FAIL + exit 1 on any error."""
    try:
        session = train_runner.run(run_name, max_steps=max_steps)
    except Exception as exc:  # noqa: BLE001 - coqui/torch/CUDA errors vary; show the type
        print(f"FAIL: train {type(exc).__name__}: {exc}")
        return 1
    print(f"PASS: run {run_name} at step {session['steps_done']} "
          f"(peak GPU memory {session['peak_gpu_memory_gb']} GB)")
    return 0


def _check_tokenizer() -> int:
    """Tokenizer round trip of the selected speaker's text; FAIL + exit 1 on error."""
    try:
        texts, n_excluded, voice = text_oov.selected_texts()
        print(f"voice {voice}: {len(texts)} clips, {n_excluded} excluded by char_map")
        text_frontend.check_tokenizer(texts, pretrained.fetch(voice)["config"])
    except (ImportError, RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: check-tokenizer: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _oov() -> int:
    """OOV report for the selected speaker's model text; FAIL + exit 1 on error."""
    try:
        text_oov.run()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: oov: {type(exc).__name__}: {exc}")
        return 1
    return 0


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


def _cluster_speakers() -> int:
    """Cluster the Drive speaker embeddings and rank the clusters; FAIL + exit 1 on error."""
    try:
        speaker_select.cluster(
            paths.speaker_emb(),
            data_audit.read_audit(paths.drive_reports() / data_audit.AUDIT_CSV))
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
        print(f"FAIL: cluster-speakers: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _save_speaker(cluster_id: int, base_voice: str) -> int:
    """Save one cluster as the training speaker; FAIL + exit 1 on error."""
    try:
        speaker_select.save_selection(cluster_id, base_voice=base_voice)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: save-speaker: {type(exc).__name__}: {exc}")
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
    elif args.command == "cluster-speakers":
        return _cluster_speakers()
    elif args.command == "save-speaker":
        return _save_speaker(args.id, args.base_voice)
    elif args.command == "oov":
        return _oov()
    elif args.command == "check-tokenizer":
        return _check_tokenizer()
    elif args.command == "prepare":
        return _prepare(args.workers)
    elif args.command == "build-config":
        return _build_config(args.run)
    elif args.command == "check-loader":
        return _check_loader()
    elif args.command == "train":
        return _train(args.run, args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
