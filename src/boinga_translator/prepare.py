"""Training-data prep: audio transform, leak-free splits, ljspeech manifests, `boinga prepare`."""

# dialect-data-rules: mono, resample to the checkpoint's rate (never change the model's
# rate), trim only leading/trailing silence keeping >= 100 ms, peak-safe normalisation.
# DECISIONS "Audio preprocessing" records the choices and their reasons.

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections.abc import Callable, Iterable
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from boinga_translator import (
    corpus,
    data_audit,
    paths,
    report_io,
    settings,
    speaker_select,
    text_frontend,
)

PAD_S = 0.1  # silence kept before the first and after the last non-silent frame
TRIM_TOP_DB = data_audit.TRIM_TOP_DB  # same silence threshold as the audit (40 dB below peak)
PEAK_DBFS = -1.0
RES_TYPE = "soxr_hq"
HELD_OUT_FRACTION = 0.05  # test (if sampled) and val, each
MIN_HELD_OUT = 20  # clips; also the least official-test clips that count as a usable test set
SPLIT_NAMES = ("train", "val", "test")
SPLITS_SUMMARY_REL = "configs/splits_summary.json"


def target_sample_rate(selection_path: Path | None = None, facts_path: Path | None = None) -> int:
    """The base voice's sample rate: configs/speaker_selection.json -> pretrained_facts.json."""
    sel = json.loads((selection_path or paths.REPO / speaker_select.SELECTION_REL).read_text(
        encoding="utf-8"))
    facts = json.loads((facts_path or paths.configs_dir() / data_audit.FACTS_FILE).read_text(
        encoding="utf-8"))
    voice = sel.get("base_voice")
    if voice not in facts:
        raise ValueError(f"base voice {voice!r} not in pretrained facts; have {sorted(facts)}")
    rate = facts[voice].get("sample_rate")
    if type(rate) is not int or rate <= 0:
        raise ValueError(f"pretrained facts give {voice!r} no valid sample_rate: {rate!r}")
    return rate


def transform_audio(in_wav: Path | str, out_wav: Path | str, target_sr: int) -> dict[str, Any]:
    """Write in_wav as a mono 16-bit wav at target_sr, edge silence trimmed, peak at -1 dBFS.

    Returns in_sr, out_sr, in_dur, out_dur (s), trimmed_s (edge silence cut, at target_sr),
    gain_db and upsampled (in_sr < target_sr). An empty, non-finite or silent clip raises
    ValueError and nothing is written; out_wav appears only when complete.
    """
    from librosa import resample  # heavy (numba): imported on first use
    from librosa.effects import trim

    data, in_sr = sf.read(str(in_wav), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError(f"{in_wav}: empty clip")
    if not np.isfinite(data).all():
        raise ValueError(f"{in_wav}: non-finite samples (NaN/inf)")
    mono = data.mean(axis=1)
    y = mono if in_sr == target_sr else resample(
        mono, orig_sr=in_sr, target_sr=target_sr, res_type=RES_TYPE)
    if not np.any(y):
        raise ValueError(f"{in_wav}: silent clip (no non-zero sample)")
    _, (start, end) = trim(y, top_db=TRIM_TOP_DB)
    pad = round(PAD_S * target_sr)
    kept = y[max(0, start - pad):min(y.size, end + pad)]
    gain = 10 ** (PEAK_DBFS / 20) / float(np.abs(kept).max())
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        sf.write(str(tmp), kept * gain, target_sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)  # no half-written file for a later "skip done" check
        raise
    return {"in_sr": int(in_sr), "out_sr": int(target_sr), "in_dur": mono.size / in_sr,
            "out_dur": kept.size / target_sr, "trimmed_s": (y.size - kept.size) / target_sr,
            "gain_db": float(20 * np.log10(gain)), "upsampled": in_sr < target_sr}


# --- splits ----------------------------------------------------------------------------------


def _held_out(n: int) -> int:
    return max(MIN_HELD_OUT, round(HELD_OUT_FRACTION * n))


def _sample(ids: pd.Series, n: int, rng: np.random.Generator) -> set[str]:
    """n ids drawn without replacement; the input order does not matter (sorted first)."""
    pool = sorted(ids)
    return {pool[i] for i in rng.permutation(len(pool))[:n]}


def make_splits(index_df: pd.DataFrame, selection_ids: Iterable[str],
                seed: int = settings.SEED) -> pd.DataFrame:
    """One row per kept clip: clip_id, split (train/val/test), duration_s.

    index_df needs clip_id, ctg_text_nfc, split_official and duration_s. Only selected clips
    are kept; clips sharing a ctg_text_nfc keep the longest one (ties: lowest clip_id), so no
    transcript reaches two splits. Test = the official test clips if >= MIN_HELD_OUT remain,
    else a seeded HELD_OUT_FRACTION sample (min MIN_HELD_OUT); val = a seeded
    HELD_OUT_FRACTION (min MIN_HELD_OUT) of the rest; train = what is left. attrs carry
    seed, test_source, n_selected and dropped_duplicate_text for save_splits.
    """
    need = {"clip_id", "ctg_text_nfc", "split_official", "duration_s"}
    if missing := need - set(index_df.columns):
        raise ValueError(f"index_df lacks column(s) {sorted(missing)}")
    wanted = pd.Index(pd.unique(pd.Series(list(selection_ids), dtype=str)))
    df = index_df.assign(clip_id=index_df["clip_id"].astype(str)).set_index("clip_id")
    if unknown := sorted(set(wanted) - set(df.index)):
        raise ValueError(f"{len(unknown)} selected clip(s) not in the index: {unknown[:5]}")
    df = df.loc[wanted].reset_index()
    order = df["clip_id"].str.zfill(12)
    df = (df.assign(_o=order).sort_values(["duration_s", "_o"], ascending=[False, True])
          .drop_duplicates("ctg_text_nfc").sort_values("_o"))
    rng = np.random.default_rng(seed)
    official = df.loc[df["split_official"] == "test", "clip_id"]
    if len(official) >= MIN_HELD_OUT:
        test, source = set(official), "official"
    else:
        test, source = _sample(df["clip_id"], _held_out(len(df)), rng), "sampled"
    rest = df.loc[~df["clip_id"].isin(test), "clip_id"]
    val = _sample(rest, _held_out(len(rest)), rng)
    if len(rest) <= len(val):
        raise ValueError(f"no clips left for train: {len(df)} unique texts, "
                         f"{len(test)} test, {len(val)} val")
    split = np.where(df["clip_id"].isin(test), "test",
                     np.where(df["clip_id"].isin(val), "val", "train"))
    out = pd.DataFrame({"clip_id": df["clip_id"].to_numpy(), "split": split,
                        "duration_s": df["duration_s"].astype(float).to_numpy()})
    by_text = pd.Series(split, index=df["ctg_text_nfc"].to_numpy()).groupby(level=0).nunique()
    if (by_text > 1).any():  # not `assert`: the guarantee must hold under python -O too
        raise RuntimeError(f"transcript(s) in two splits: {list(by_text[by_text > 1].index[:3])}")
    out.attrs.update(seed=seed, test_source=source, n_selected=len(wanted),
                     dropped_duplicate_text=len(wanted) - len(df))
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_splits(df: pd.DataFrame, csv_path: Path | None = None,
                summary_path: Path | None = None) -> dict[str, Any]:
    """Write splits.csv (Drive, paths.splits_csv()) and configs/splits_summary.json.

    Refuses to replace a saved splits.csv whose test clips differ (the test set is fixed
    once and never trained on). Returns the summary.
    """
    if not {"seed", "test_source", "dropped_duplicate_text"} <= set(df.attrs):
        raise ValueError("save_splits needs the frame make_splits returns (attrs missing)")
    csv_path = csv_path or paths.splits_csv()
    summary_path = summary_path or paths.REPO / SPLITS_SUMMARY_REL
    new_test = set(df.loc[df["split"] == "test", "clip_id"].astype(str))
    if csv_path.is_file():
        old = pd.read_csv(csv_path, dtype={"clip_id": str})
        if set(old.loc[old["split"] == "test", "clip_id"]) != new_test:
            raise ValueError(f"{csv_path} already fixes a different test set; refusing to "
                             "change it (delete it on purpose to re-split)")
    data = df[["clip_id", "split", "duration_s"]].to_csv(index=False, lineterminator="\n")
    blob = data.encode("utf-8")
    _atomic_write(csv_path, blob)
    counts = df["split"].value_counts()
    minutes = df.groupby("split")["duration_s"].sum() / 60
    summary = {
        "counts": {k: int(counts.get(k, 0)) for k in SPLIT_NAMES},
        "minutes": {k: round(float(minutes.get(k, 0.0)), 2) for k in SPLIT_NAMES},
        "csv_sha256": hashlib.sha256(blob).hexdigest(),
        "seed": int(df.attrs["seed"]),
        "test_source": df.attrs["test_source"],
        "dropped_duplicate_text": int(df.attrs["dropped_duplicate_text"]),
    }
    _atomic_write(summary_path, (json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return summary


# --- manifests -------------------------------------------------------------------------------
# coqui's "ljspeech" formatter (coqui-tts 0.28.0) reads <root>/<meta_file>, splits each line on
# "|", takes column 0 as the wav id (<root>/wavs/<id>.wav) and column 2 as the text, without
# stripping: the last column keeps the line's "\n", which the tokenizer drops as an unknown
# character (DECISIONS "Training manifests"). Names come from pretrained_facts.json "dataset".

FORMATTER = "ljspeech"
Samples = list[dict[str, Any]]
Loader = Callable[[Path, str, str | None], tuple[Samples, Samples]]


def manifest_names(facts_path: Path | None = None) -> dict[str, str]:
    """{split: meta file name} from configs/pretrained_facts.json "dataset" (ljspeech only)."""
    facts = json.loads((facts_path or paths.configs_dir() / data_audit.FACTS_FILE).read_text(
        encoding="utf-8"))
    ds = facts.get("dataset")
    if not isinstance(ds, dict) or ds.get("formatter") != FORMATTER:
        raise ValueError(f"pretrained facts need dataset.formatter {FORMATTER!r}, got {ds!r}")
    names = ds.get("meta_files")
    if not isinstance(names, dict) or set(names) != set(SPLIT_NAMES):
        raise ValueError(f"pretrained facts dataset.meta_files must name {SPLIT_NAMES}: "
                         f"{names!r}")
    return {k: str(names[k]) for k in SPLIT_NAMES}


def manifest_wav(root: Path, clip_id: str) -> Path:
    """Where the ljspeech formatter looks for a clip's audio: <root>/wavs/<clip_id>.wav."""
    return root / "wavs" / f"{clip_id}.wav"


def write_manifests(splits_df: pd.DataFrame, index_df: pd.DataFrame,
                    out_root: Path | None = None,
                    facts_path: Path | None = None) -> dict[str, Path]:
    """Write <out_root>/<meta file> per split: "<clip_id>|<text>|<text>\\n", UTF-8, no header.

    text = text_frontend.prepare_text(ctg_text): the verbatim transcript plus char_map only.
    Refuses, before writing anything, a clip with no transcript, an empty model text, or a
    "|", CR or LF in the id or text. Default out_root: paths.data_prepared().
    """
    out_root = Path(out_root) if out_root is not None else paths.data_prepared()
    names = manifest_names(facts_path)
    raw = index_df.assign(clip_id=index_df["clip_id"].astype(str)).set_index("clip_id")
    if raw.index.duplicated().any():
        raise ValueError("index_df lists a clip_id twice")
    if odd := set(splits_df["split"]) - set(SPLIT_NAMES):
        raise ValueError(f"unknown split name(s) {sorted(map(str, odd))}; use {SPLIT_NAMES}")
    ids = splits_df["clip_id"].astype(str)
    if missing := sorted(set(ids) - set(raw.index)):
        raise ValueError(f"{len(missing)} clip(s) have no transcript in the index: "
                         f"{missing[:5]}")
    lines: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
    for cid, split in zip(ids, splits_df["split"], strict=True):
        text = text_frontend.prepare_text(str(raw.at[cid, "ctg_text"]))
        if not text or any(c in cid + text for c in "|\r\n"):
            raise ValueError(f"clip {cid}: empty model text or '|'/newline in id or text: "
                             f"{text!r}")
        lines[str(split)].append(f"{cid}|{text}|{text}\n")
    out = {}
    for split in SPLIT_NAMES:
        out[split] = out_root / names[split]
        _atomic_write(out[split], "".join(lines[split]).encode("utf-8"))
    return out


def _coqui_load(root: Path, train: str, val: str | None) -> tuple[Samples, Samples]:
    """coqui load_tts_samples on one ljspeech dataset (0.28 wants a config with `datasets`)."""
    from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseTTSConfig
    from TTS.tts.datasets import load_tts_samples

    ds = BaseDatasetConfig(formatter=FORMATTER, path=str(root), meta_file_train=train,
                           meta_file_val=val or "")
    return load_tts_samples(BaseTTSConfig(datasets=[ds]), eval_split=val is not None)


def _check_samples(split: str, samples: Samples, manifest: Path) -> list[str]:
    """Problems with one split's parsed samples, compared with the manifest we wrote."""
    want = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        cid, text, _ = line.split("|")
        want[cid] = text
    problems = []
    if len(samples) != len(want):
        problems.append(f"{split}: parsed {len(samples)} samples, manifest has {len(want)}")
    if gone := [s["audio_file"] for s in samples if not Path(s["audio_file"]).is_file()]:
        problems.append(f"{split}: {len(gone)} audio file(s) missing, e.g. {gone[:3]}")
    # Only the line's own newline may trail the text (the formatter does not strip it).
    if bad := [s["audio_file"] for s in samples
               if want.get(Path(s["audio_file"]).stem) != s["text"].removesuffix("\n")]:
        problems.append(f"{split}: text differs from the manifest for {len(bad)} sample(s), "
                        f"e.g. {bad[:3]}")
    return problems


def parse_check(root: Path | None = None, loader: Loader | None = None,
                facts_path: Path | None = None) -> dict[str, int]:
    """Load the 3 manifests with coqui's ljspeech formatter, check them and print PASS.

    Counts must match the manifests, every audio_file must exist and every parsed text must
    equal ours (one trailing "\\n" allowed). Raises RuntimeError listing every problem.
    """
    root = Path(root) if root is not None else paths.data_prepared()
    names = manifest_names(facts_path)
    load = loader or _coqui_load
    train, val = load(root, names["train"], names["val"])
    test, _ = load(root, names["test"], None)
    parsed = {"train": train, "val": val, "test": test}
    for split in SPLIT_NAMES:
        print(f"{split}: {len(parsed[split])} samples from {names[split]}")
    if train:
        print(f"first train sample: text={train[0]['text']!r} "
              f"audio_file={train[0]['audio_file']}")
    problems = [p for k in SPLIT_NAMES for p in _check_samples(k, parsed[k], root / names[k])]
    if problems:
        raise RuntimeError("; ".join(problems))
    print(f"PASS: coqui parsed {len(train)} train / {len(val)} val samples "
          f"(+ {len(test)} test)")
    return {k: len(v) for k, v in parsed.items()}


# --- build: `boinga prepare` -----------------------------------------------------------------
# The fast disk (paths.data_prepared()) is lost with the VM; Drive keeps one tar per splits
# sha256 (paths.prepared_backup()). A fresh VM restores that tar instead of recomputing.

STATS_CSV = "clip_stats.csv"  # one row per clip of the splits: transform stats or its error
Job = tuple[str, str, str, int]  # clip_id, source wav, prepared wav, target rate


def _transform_job(job: Job) -> dict[str, Any]:
    """transform_audio for one clip; a bad clip gives an error row, never raises (pool-safe)."""
    cid, src, dst, sr = job
    try:
        return {"clip_id": cid, **transform_audio(src, dst, sr), "error": ""}
    except Exception as exc:  # noqa: BLE001 - one bad clip (librosa ParameterError too) must not kill the pool
        return {"clip_id": cid, "error": f"{type(exc).__name__}: {exc}"}


def _transform_all(jobs: list[Job], root: Path, workers: int) -> pd.DataFrame:
    """Stats for every job; a clip with a stats row and its wav (or an error) is skipped."""
    csv = root / STATS_CSV
    old = (pd.read_csv(csv, dtype={"clip_id": str, "error": str}, keep_default_na=False)
           if csv.is_file() else pd.DataFrame(columns=["clip_id", "error"]))
    kept = [e != "" or manifest_wav(root, c).is_file() for c, e in zip(
        old["clip_id"], old["error"], strict=True)]
    done = old[pd.Series(kept, index=old.index, dtype=bool) & old["clip_id"].isin(
        {j[0] for j in jobs})]
    todo = [j for j in jobs if j[0] not in set(done["clip_id"])]
    print(f"transform: {len(todo)} clip(s) to do, {len(done)} already done")
    if workers > 1 and len(todo) > 1:
        with Pool(workers) as pool:
            rows = list(tqdm(pool.imap_unordered(_transform_job, todo, chunksize=16),
                             total=len(todo)))
    else:
        rows = [_transform_job(j) for j in tqdm(todo)]
    stats = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    order = {j[0]: i for i, j in enumerate(jobs)}
    stats = stats.sort_values("clip_id", key=lambda s: s.map(order)).reset_index(drop=True)
    stats["error"] = stats["error"].fillna("")
    _atomic_write(csv, stats.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return stats


def _backup_tar(splits_sha: str) -> Path:
    """The Drive tar of the prepared folder for one splits.csv sha256."""
    return paths.prepared_backup() / f"prepared-{splits_sha[:16]}.tar"


def _report(stats: pd.DataFrame, splits: pd.DataFrame, files: dict[str, Path]) -> None:
    ok = stats[stats["error"] == ""].merge(splits[["clip_id", "split"]], on="clip_id")
    for split in SPLIT_NAMES:
        part = ok[ok["split"] == split]
        print(f"{split}: {len(part)} clips, {part['out_dur'].astype(float).sum() / 60:.2f} min")
    print(f"upsampled: {int(ok['upsampled'].astype(str).eq('True').sum())} of {len(ok)} clips; "
          f"mean trimmed {ok['trimmed_s'].astype(float).mean():.3f} s")
    bad = stats.loc[stats["error"] != "", "error"]
    print(f"failed: {len(bad)} clip(s), by reason: "
          f"{bad.str.split(':').str[0].value_counts().to_dict()}")
    for split in SPLIT_NAMES:
        print(f"--- {files[split].name} (first 3 lines)")
        print("".join(files[split].read_text(encoding="utf-8").splitlines(True)[:3]), end="")


def build(workers: int = 2) -> dict[str, int]:
    """`boinga prepare`: splits, 22.05 kHz wavs, manifests and a Drive tar; PASS line.

    The selection is the C4 speaker (sha-checked) minus char_map-excluded rows. Each
    clip is transformed once (skipped if done); a fresh VM restores the Drive tar made for
    the same splits sha256. Clips whose transform fails are counted and left out.
    Returns {"clips": clips in the manifests, "failed": clips left out}.
    """
    index = corpus.load_index()
    selection, _ = speaker_select.load_selection()
    text = index.set_index("clip_id")["ctg_text_nfc"]
    ids = [c for c in selection["clip_id"] if not text_frontend.excluded(str(text.get(c, "")))]
    print(f"selection: {len(selection)} clips, {len(selection) - len(ids)} excluded by char_map")
    sr = target_sample_rate()
    splits = make_splits(index.merge(selection[["clip_id", "duration_s"]], on="clip_id",
                                     how="left"), ids)
    summary = save_splits(splits)
    root, tar = paths.data_prepared(), _backup_tar(summary["csv_sha256"])
    if tar.is_file() and not (root / STATS_CSV).is_file():
        with tarfile.open(tar) as t:
            t.extractall(root, filter="data")
        print(f"restored {root} from {tar}")
    wav = index.set_index("clip_id")["wav_path"]
    jobs = [(c, str(wav[c]), str(manifest_wav(root, c)), sr) for c in splits["clip_id"]]
    stats = _transform_all(jobs, root, workers)
    good = set(stats.loc[stats["error"] == "", "clip_id"])
    files = write_manifests(splits[splits["clip_id"].isin(good)], index, root)
    if not tar.is_file():
        tmp = tar.with_name(tar.name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp, "w") as t:
            t.add(root, arcname=".")
        os.replace(tmp, tar)
        print(f"backed up {root} to {tar}")
    _report(stats, splits, files)
    report_io.emit(paths.REPO / SPLITS_SUMMARY_REL, SPLITS_SUMMARY_REL)
    print(f"PASS: prepared {len(good)} clips")
    return {"clips": len(good), "failed": len(stats) - len(good)}


def pick_prepared(n: int = 5, seed: int = settings.SEED,
                  root: Path | None = None) -> tuple[list[str], list[str]]:
    """n seeded train clips as (prepared wav paths, "<clip_id>: <manifest text>" labels)."""
    root = Path(root) if root is not None else paths.data_prepared()
    lines = (root / manifest_names()["train"]).read_text(encoding="utf-8").splitlines()
    picked = [lines[i] for i in np.random.default_rng(seed).permutation(len(lines))[:n]]
    cols = [line.split("|") for line in picked]
    return [str(manifest_wav(root, c[0])) for c in cols], [f"{c[0]}: {c[2]}" for c in cols]
