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
    speaker_select,
)
from boinga_translator.baseline import bangla_similarity

N_OWN, N_OTHER = 15, 5  # own speaker's test clips (real reference), other speakers' test clips
WORDS = (5, 15)  # preferred word count of a prompt
PROMPTS_FILE = "eval_prompts.json"
PROMPTS_REL = f"configs/{PROMPTS_FILE}"


@dataclass(frozen=True)
class Tables:
    """Inputs of the pick; load_tables() reads them on Colab, tests pass small ones."""

    audit: pd.DataFrame  # Drive audit.csv: clip_id, split_official, ctg_text_nfc, rule columns
    index: pd.DataFrame  # corpus index: clip_id, ctg_text (verbatim), bn_text
    splits: pd.DataFrame  # Drive splits.csv: clip_id, split
    own_ids: set[str]  # the training speaker's clips (speaker_selection.csv)
    clusters: pd.DataFrame  # Drive speaker_clusters.csv: clip_id, cluster


def load_tables() -> Tables:
    """Audit, corpus index, splits, speaker selection and clusters from Drive / the fast disk."""
    selection, _ = speaker_select.load_selection()
    return Tables(audit=data_audit.read_audit(paths.drive_reports() / data_audit.AUDIT_CSV),
                  index=corpus.load_index(),
                  splits=pd.read_csv(paths.splits_csv(), dtype={"clip_id": str}),
                  own_ids=set(selection["clip_id"].astype(str)),
                  clusters=pd.read_csv(paths.speaker_clusters(), dtype={"clip_id": str}))


def _ranked(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """5-15 words first, then least like the Bangla translation; ties by a seeded shuffle."""
    df = df.sample(frac=1, random_state=seed)
    sim = [bangla_similarity(c, b) for c, b in zip(df["ctg_text_nfc"], df["bn_text"], strict=True)]
    out = ~df["ctg_text_nfc"].str.split().str.len().between(*WORDS)
    return df.assign(out=out, sim=sim).sort_values(["out", "sim"], kind="stable")


def _clusters(tables: Tables) -> pd.Series:
    """clip_id -> voice cluster label."""
    return tables.clusters.astype({"clip_id": str}).set_index("clip_id")["cluster"]


def candidates(tables: Tables, seed: int = settings.SEED) -> tuple[list[str], list[str]]:
    """(own-speaker test clip ids, other-speaker official-test clip ids), best first.

    Clean clips with a translation only; no text that a train/val clip has. Other speakers:
    not in the selection, its voice clusters or splits.csv; one clip per cluster first.
    """
    audit = tables.audit.astype({"clip_id": str})
    split = tables.splits.astype({"clip_id": str}).set_index("clip_id")["split"]
    seen = set(audit.loc[audit["clip_id"].map(split).isin(["train", "val"]), "ctg_text_nfc"])
    clean = audit[audit_summary.clean_mask(audit) & ~audit["ctg_text_nfc"].isin(seen)]
    clean = clean.merge(tables.index[["clip_id", "bn_text"]].astype({"clip_id": str}),
                        on="clip_id")
    clean = clean[clean["bn_text"].fillna("").astype(str).str.strip() != ""]
    cluster = _clusters(tables)
    own_clusters = {cluster[c] for c in tables.own_ids if c in cluster}
    own = _ranked(clean[clean["clip_id"].map(split).eq("test")], seed)
    own = own.drop_duplicates("ctg_text_nfc")
    other = clean[clean["split_official"].eq("test") & ~clean["clip_id"].isin(tables.own_ids)
                  & ~clean["clip_id"].isin(split.index)
                  & ~clean["clip_id"].map(cluster).isin(own_clusters)
                  & ~clean["ctg_text_nfc"].isin(own["ctg_text_nfc"])]
    other = _ranked(other, seed).drop_duplicates("ctg_text_nfc")
    rounds = other.groupby(other["clip_id"].map(cluster).fillna(-1)).cumcount()
    other = other.assign(round=rounds).sort_values("round", kind="stable")
    return list(own["clip_id"]), list(other["clip_id"])


def pick_prompts(seed: int = settings.SEED, exclude_ids: Iterable[str] = (),
                 tables: Tables | None = None, previous: Path | None = None) -> list[dict[str, Any]]:
    """P01-P15 own-speaker test clips (has_reference) + P16-P20 other speakers' clips.

    Without configs/eval_prompts.json: a fresh pick. With it: its prompts are kept, and each
    id in exclude_ids gets the next-best unused clip of its group; a replaced clip is listed
    in the entry's "replaced" and never comes back. An other-speaker replacement comes from a
    voice cluster no kept prompt uses, when there is one.
    """
    path = previous or paths.configs_dir() / PROMPTS_FILE
    exclude = set(exclude_ids)
    tables = tables if tables is not None else load_tables()
    own, other = candidates(tables, seed)
    meta = tables.index.astype({"clip_id": str}).set_index("clip_id")

    def entry(pid: str, cid: str, ref: bool, replaced: list[str] | None = None) -> dict[str, Any]:
        e = {"id": pid, "clip_id": cid, "ctg_text": str(meta.at[cid, "ctg_text"]),
             "bn_text": str(meta.at[cid, "bn_text"]), "has_reference": ref}
        return {**e, "replaced": replaced} if replaced else e

    if not path.is_file():
        if exclude:
            raise ValueError(f"no {PROMPTS_REL} to replace prompts in: pick without --exclude")
        for name, pool, n in (("own-speaker test", own, N_OWN), ("other-speaker", other, N_OTHER)):
            if len(pool) < n:
                raise ValueError(f"need {n} {name} candidates, found {len(pool)}")
        ids = own[:N_OWN] + other[:N_OTHER]
        return [entry(f"P{i:02d}", cid, i <= N_OWN) for i, cid in enumerate(ids, 1)]
    old = json.loads(path.read_text(encoding="utf-8"))
    if unknown := sorted(exclude - {p["id"] for p in old}):
        raise ValueError(f"unknown prompt id(s) {', '.join(unknown)} in {PROMPTS_REL}")
    if stale := [p["id"] for p in old if p["clip_id"] not in {*own, *other}]:
        raise ValueError(f"{', '.join(stale)} in {PROMPTS_REL} are no longer candidates: "
                         "delete the file to pick afresh")
    print(f"kept {PROMPTS_REL}" + (f", replacing {', '.join(sorted(exclude))}" if exclude else ""))
    cluster = _clusters(tables)
    taken = {cluster.get(p["clip_id"]) for p in old
             if not p["has_reference"] and p["id"] not in exclude}
    used = {p["clip_id"] for p in old} | {c for p in old for c in p.get("replaced", [])}
    out = []
    for p in old:
        if p["id"] not in exclude:
            out.append(p)
            continue
        pool = [c for c in (own if p["has_reference"] else other) if c not in used]
        if not pool:
            raise ValueError(f"no candidate left to replace {p['id']}")
        nxt = pool[0] if p["has_reference"] else next(
            (c for c in pool if cluster.get(c) not in taken), pool[0])
        used.add(nxt)
        taken |= set() if p["has_reference"] else {cluster.get(nxt)}
        out.append(entry(p["id"], nxt, p["has_reference"], [*p.get("replaced", []), p["clip_id"]]))
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
        ref = "own speaker, has reference" if p["has_reference"] else "other speaker"
        print(f"{p['id']} ({ref}, clip {p['clip_id']})\n  ctg: {p['ctg_text']}\n"
              f"  bn:  {p['bn_text']}")
