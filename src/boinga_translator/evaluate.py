"""Blind A/B evaluation (C10): 20 held-out, dialect-heavy prompts for the listening test."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from boinga_translator import (
    audit_summary,
    corpus,
    data_audit,
    paths,
    report_io,
    settings,
)
from boinga_translator.baseline import bangla_similarity

N_PROMPTS = 20  # all from our test split: the corpus has too few other-speaker clips (DEBUG_LOG)
WORDS = (5, 15)  # preferred word count of a prompt
PROMPTS_FILE = "eval_prompts.json"
PROMPTS_REL = f"configs/{PROMPTS_FILE}"


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
