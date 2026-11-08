"""Blind A/B evaluation (C10): 20 held-out, dialect-heavy prompts for the listening test."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from boinga_translator import (
    audit_summary,
    corpus,
    data_audit,
    paths,
    prepare,
    report_io,
    settings,
)
from boinga_translator.baseline import bangla_similarity, rms_dbfs

N_PROMPTS = 20  # all from our test split: the corpus has too few other-speaker clips (DEBUG_LOG)
WORDS = (5, 15)  # preferred word count of a prompt
PROMPTS_FILE = "eval_prompts.json"
PROMPTS_REL = f"configs/{PROMPTS_FILE}"
VOICE_SETS = {"A_baseline": "baseline", "B_boinga": "boinga"}  # set folder -> voice
REF_SET = "R_real"
INDEX_COLS = ["id", "set", "path", "duration_s", "rms_db", "text"]
GEN_REL = "reports/eval_generation.md"
MIN_SECONDS, MIN_DBFS = 0.5, -50.0  # same floor as the C9 reload check


@dataclass(frozen=True)
class Tables:
    """Inputs of the pick; load_tables() reads them on Colab, tests pass small ones."""

    audit: pd.DataFrame  # Drive audit.csv: clip_id, split_official, ctg_text_nfc, rule columns
    index: pd.DataFrame  # corpus index: clip_id, ctg_text (verbatim), bn_text
    splits: pd.DataFrame  # Drive splits.csv: clip_id, split


def load_tables() -> Tables:
    """Audit and splits from Drive, the corpus index from the fast disk."""
    return Tables(audit=data_audit.read_audit(paths.drive_reports() / data_audit.AUDIT_CSV),
                  index=corpus.load_index(),
                  splits=pd.read_csv(paths.splits_csv(), dtype={"clip_id": str}))


def _ranked(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """5-15 words first, then least like the Bangla translation; ties by a seeded shuffle."""
    df = df.sample(frac=1, random_state=seed)
    sim = [bangla_similarity(c, b) for c, b in zip(df["ctg_text_nfc"], df["bn_text"], strict=True)]
    out = ~df["ctg_text_nfc"].str.split().str.len().between(*WORDS)
    return df.assign(out=out, sim=sim).sort_values(["out", "sim"], kind="stable")


def candidates(tables: Tables, seed: int = settings.SEED) -> list[str]:
    """Clip ids of our test split, best first.

    Clean clips with a translation only; distinct text; no text that a train/val clip has.
    """
    audit = tables.audit.astype({"clip_id": str})
    split = tables.splits.astype({"clip_id": str}).set_index("clip_id")["split"]
    seen = set(audit.loc[audit["clip_id"].map(split).isin(["train", "val"]), "ctg_text_nfc"])
    test = audit[audit_summary.clean_mask(audit) & audit["clip_id"].map(split).eq("test")
                 & ~audit["ctg_text_nfc"].isin(seen)]
    test = test.merge(tables.index[["clip_id", "bn_text"]].astype({"clip_id": str}),
                      on="clip_id")
    test = test[test["bn_text"].fillna("").astype(str).str.strip() != ""]
    return list(_ranked(test, seed).drop_duplicates("ctg_text_nfc")["clip_id"])


def pick_prompts(seed: int = settings.SEED, exclude_ids: Iterable[str] = (),
                 tables: Tables | None = None, previous: Path | None = None) -> list[dict[str, Any]]:
    """P01-P20 from our test split, each with its real recording as the reference.

    Without configs/eval_prompts.json: a fresh pick. With it: its prompts are kept, and each
    id in exclude_ids gets the next-best unused clip; a replaced clip is listed in the
    entry's "replaced" and never comes back.
    """
    path = previous or paths.configs_dir() / PROMPTS_FILE
    exclude = set(exclude_ids)
    tables = tables if tables is not None else load_tables()
    pool = candidates(tables, seed)
    meta = tables.index.astype({"clip_id": str}).set_index("clip_id")

    def entry(pid: str, cid: str, replaced: list[str] | None = None) -> dict[str, Any]:
        e = {"id": pid, "clip_id": cid, "ctg_text": str(meta.at[cid, "ctg_text"]),
             "bn_text": str(meta.at[cid, "bn_text"]), "has_reference": True}
        return {**e, "replaced": replaced} if replaced else e

    if not path.is_file():
        if exclude:
            raise ValueError(f"no {PROMPTS_REL} to replace prompts in: pick without --exclude")
        if len(pool) < N_PROMPTS:
            raise ValueError(f"need {N_PROMPTS} test-split candidates, found {len(pool)}")
        return [entry(f"P{i:02d}", cid) for i, cid in enumerate(pool[:N_PROMPTS], 1)]
    old = json.loads(path.read_text(encoding="utf-8"))
    if unknown := sorted(exclude - {p["id"] for p in old}):
        raise ValueError(f"unknown prompt id(s) {', '.join(unknown)} in {PROMPTS_REL}")
    in_pool = set(pool)
    if stale := [p["id"] for p in old if p["clip_id"] not in in_pool]:
        raise ValueError(f"{', '.join(stale)} in {PROMPTS_REL} are no longer candidates: "
                         "delete the file to pick afresh")
    print(f"kept {PROMPTS_REL}" + (f", replacing {', '.join(sorted(exclude))}" if exclude else ""))
    used = {p["clip_id"] for p in old} | {c for p in old for c in p.get("replaced", [])}
    free = iter([c for c in pool if c not in used])
    out = []
    for p in old:
        if p["id"] not in exclude:
            out.append(p)
            continue
        if (nxt := next(free, None)) is None:
            raise ValueError(f"no candidate left to replace {p['id']}")
        out.append(entry(p["id"], nxt, [*p.get("replaced", []), p["clip_id"]]))
    return out


def write_prompts(prompts: list[dict[str, Any]]) -> Path:
    """Write configs/eval_prompts.json (repo + Drive copy) and emit it for the repo."""
    text = json.dumps(prompts, ensure_ascii=False, indent=2) + "\n"
    repo_path, _ = report_io.write_report(text, PROMPTS_REL)
    report_io.emit(repo_path, PROMPTS_REL)
    return repo_path


def show_prompts(path: Path | None = None) -> None:
    """Print each prompt's id, Chittagonian text and Bangla translation for review."""
    for p in json.loads((path or paths.configs_dir() / PROMPTS_FILE).read_text(encoding="utf-8")):
        print(f"{p['id']} (clip {p['clip_id']})\n  ctg: {p['ctg_text']}\n"
              f"  bn:  {p['bn_text']}")

# --- C10.2: evaluation audio -----------------------------------------------------------------


def final_dir() -> Path:
    """Drive folder of the evaluation audio: A_baseline/, B_boinga/, R_real/ and index.csv."""
    return paths.eval_dir() / "final"


def load_prompts() -> list[dict[str, Any]]:
    """The reviewed prompts of configs/eval_prompts.json."""
    return json.loads((paths.configs_dir() / PROMPTS_FILE).read_text(encoding="utf-8"))


def _bank() -> Any:
    from boinga_translator import synth  # coqui is imported only when a voice is loaded

    return synth.VoiceBank()


def _corpus_wavs() -> dict[str, Path]:
    index = corpus.load_index()
    return {str(c): Path(w) for c, w in zip(index["clip_id"], index["wav_path"], strict=True)
            if isinstance(w, str) and w}


def _target_rate() -> int:
    return prepare.target_sample_rate()


def _checked(wav: np.ndarray, rate: int, what: str) -> tuple[float, float]:
    """(seconds, RMS dBFS) of a wav; ValueError if too short, too quiet or not finite."""
    if wav.size == 0:
        raise ValueError(f"{what}: empty wav")
    seconds, level = wav.size / rate, rms_dbfs(wav)
    if not (seconds > MIN_SECONDS and level > MIN_DBFS and np.isfinite(wav).all()):
        raise ValueError(f"{what}: {seconds:.2f} s at {level:.1f} dBFS is too short, quiet or "
                         f"not finite (needs > {MIN_SECONDS} s and > {MIN_DBFS} dBFS)")
    return seconds, level


def _one(p: dict[str, Any], name: str, path: Path, bank: Any,
         wav_paths: dict[str, Path]) -> dict[str, Any]:
    """Make one wav of one set and return its index row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == REF_SET:
        if p["clip_id"] not in wav_paths:
            raise KeyError(f"no corpus wav for clip {p['clip_id']}")
        prepare.transform_audio(wav_paths[p["clip_id"]], path, _target_rate())
        wav, rate = sf.read(path, dtype="float32")
        seconds, level = _checked(np.asarray(wav), rate, f"{p['id']} {name}")
    else:
        rate, wav = bank.speak(VOICE_SETS[name], p["ctg_text"])
        seconds, level = _checked(np.asarray(wav), rate, f"{p['id']} {name}")
        sf.write(path, wav, rate)  # the raw model output, not normalised
    return {"id": p["id"], "set": name, "path": str(path), "duration_s": round(seconds, 3),
            "rms_db": round(level, 1), "text": p["ctg_text"]}


def generate(bank: Any = None, wav_paths: dict[str, Path] | None = None) -> dict[str, int]:
    """Speak every prompt as A_baseline and B_boinga, and put each reference clip in R_real.

    Voices use VoiceBank's fixed seed and inference scales; R is the corpus clip through
    prepare.transform_audio (mono, model rate, edge silence trimmed). A wav already listed in
    index.csv with the same text is kept. Writes final/index.csv and reports/eval_generation.md;
    returns {"A": n, "B": n, "R": n}.
    """
    prompts, out = load_prompts(), final_dir()
    csv = out / "index.csv"
    old = (pd.read_csv(csv, dtype=str, keep_default_na=False) if csv.is_file()
           else pd.DataFrame(columns=INDEX_COLS))
    done = {(r["id"], r["set"]): r for r in old.to_dict("records") if Path(r["path"]).is_file()}
    bank = bank if bank is not None else _bank()
    for voice in VOICE_SETS.values():
        if voice not in bank.available():
            raise ValueError(f"voice {voice!r} not available: run `boinga promote` / "
                             "`boinga fetch-pretrained` first")
    if wav_paths is None:
        wav_paths = _corpus_wavs() if any(p["has_reference"] for p in prompts) else {}
    rows: list[dict[str, Any]] = []
    for p in tqdm(prompts, desc="eval audio", unit="prompt"):
        for name in [*VOICE_SETS, *([REF_SET] if p["has_reference"] else [])]:
            prev = done.get((p["id"], name))
            if prev is not None and prev["text"] == p["ctg_text"]:
                rows.append(prev)
            else:
                rows.append(_one(p, name, out / name / f"{p['id']}.wav", bank, wav_paths))
    index = pd.DataFrame(rows, columns=INDEX_COLS)
    index.to_csv(csv, index=False, lineterminator="\n")
    _report(index)
    return {name[0]: int((index["set"] == name).sum()) for name in [*VOICE_SETS, REF_SET]}


def _report(index: pd.DataFrame) -> None:
    num = index.astype({"duration_s": float, "rms_db": float})
    lines = ["# Evaluation audio", "",
             (f"{index['id'].nunique()} prompts (configs/eval_prompts.json), audio on Drive "
              "eval/final/. Voices use the fixed seed and inference scales; R_real is the "
              "corpus clip through prepare.transform_audio."), "",
             "| set | wavs | mean duration s | min s | max s | mean RMS dBFS |",
             "|---|---|---|---|---|---|"]
    for name, g in num.groupby("set"):
        lines.append(f"| {name} | {len(g)} | {g['duration_s'].mean():.2f} | "
                     f"{g['duration_s'].min():.2f} | {g['duration_s'].max():.2f} | "
                     f"{g['rms_db'].mean():.1f} |")
    repo_path, _ = report_io.write_report("\n".join(lines) + "\n", GEN_REL)
    report_io.emit(repo_path, GEN_REL)


def players(ids: Iterable[str] = ("P01", "P02", "P03")) -> tuple[list[str], list[str]]:
    """(wav paths, labels) of these prompts' A/B/R wavs from final/index.csv, for play()."""
    index = pd.read_csv(final_dir() / "index.csv", dtype=str, keep_default_na=False)
    ids = list(ids)
    if missing := [i for i in ids if i not in set(index["id"])]:
        raise KeyError(f"prompt id(s) {', '.join(missing)} not in {final_dir() / 'index.csv'}")
    rows = [r for i in ids for r in index[index["id"] == i].sort_values("set").to_dict("records")]
    return [r["path"] for r in rows], [f"{r['id']} {r['set']}: {r['text']}" for r in rows]
