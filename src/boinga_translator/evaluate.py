"""Blind A/B evaluation (C10): held-out prompts, A/B/R audio, blind rating sheet, scoring."""

from __future__ import annotations

import hashlib
import json
import re
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

# --- C10.3: blind rating sheet ---------------------------------------------------------------

BLIND_RMS_DBFS = -23.0  # every blind clip is scaled to this RMS, so level never gives a voice away
PEAK_MAX = 0.99
SHEET_COLS = ["rater", "prompt_id", "clip", "text", "intelligibility", "naturalness",
              "dialect_authenticity", "dialect_words", "comments"]
INSTRUCTIONS_REL = "reports/rater_instructions.md"
INSTRUCTIONS = """\
# Rating instructions / রেটিং নির্দেশিকা

## English

You will hear 20 short sentences in Chittagonian. Each sentence is spoken by two computer
voices, in `<id>_clip1.wav` and `<id>_clip2.wav`. The order is random and you are not told
which voice is which: please do not try to guess. Read the sentence in `rating_sheet.csv`,
listen to each clip as often as you like, then fill in its row:

- `rater`: your name or initials (the same in every row).
- `intelligibility` (1-5): how easily you understand the words. 1 = nothing, 5 = every word.
- `naturalness` (1-5): how human and fluent it sounds. 1 = very robotic or broken,
  5 = like a real person.
- `dialect_authenticity` (1-5): how much it sounds like a native Chittagonian speaker.
  1 = not at all (e.g. standard Bangla or a foreign accent), 5 = a native speaker.
- `dialect_words` (1-5): are the Chittagonian-specific words said correctly?
  1 = mostly wrong, 5 = all correct.
- `comments` (optional): anything you noticed, e.g. a wrong word, noise, speed.

Write the scores as the digits 1, 2, 3, 4 or 5. Rate each clip on its own. Please do not
share the files or your ratings with other raters until everyone has finished.

## বাংলা

আপনি চাটগাঁইয়া ভাষায় ২০টি ছোট বাক্য শুনবেন। প্রতিটি বাক্য দুটি কম্পিউটার কণ্ঠে বলা
হয়েছে: `<id>_clip1.wav` এবং `<id>_clip2.wav`। ক্রম এলোমেলো, আর কোনটি কোন কণ্ঠ তা
জানানো হবে না: দয়া করে অনুমান করার চেষ্টা করবেন না। `rating_sheet.csv`-এ বাক্যটি পড়ুন, প্রতিটি
ক্লিপ যতবার ইচ্ছা শুনুন, তারপর তার সারিটি পূরণ করুন:

- `rater`: আপনার নাম বা নামের আদ্যক্ষর (প্রতিটি সারিতে একই)।
- `intelligibility` (1-5): শব্দগুলো কত সহজে বোঝা যায়। 1 = কিছুই না, 5 = প্রতিটি শব্দ।
- `naturalness` (1-5): কতটা মানুষের মতো ও সাবলীল শোনায়। 1 = খুব যান্ত্রিক বা ভাঙা,
  5 = সত্যিকারের মানুষের মতো।
- `dialect_authenticity` (1-5): কতটা একজন স্থানীয় চাটগাঁইয়া ভাষীর মতো শোনায়।
  1 = একেবারেই না (যেমন প্রমিত বাংলা বা বিদেশি টান), 5 = স্থানীয় ভাষীর মতো।
- `dialect_words` (1-5): চাটগাঁইয়া-নির্দিষ্ট শব্দগুলো কি ঠিকভাবে বলা হয়েছে?
  1 = বেশিরভাগ ভুল, 5 = সবগুলো ঠিক।
- `comments` (ঐচ্ছিক): যা লক্ষ করেছেন, যেমন ভুল শব্দ, নয়েজ, গতি।

নম্বরগুলো 1, 2, 3, 4 বা 5 অঙ্কে লিখুন। প্রতিটি ক্লিপ আলাদাভাবে মূল্যায়ন করুন। সবার
রেটিং শেষ না হওয়া পর্যন্ত অন্য রেটারদের সাথে ফাইল বা রেটিং শেয়ার করবেন না।

---
Sentences: "A Synchronized Chittagonian Speech Corpus" (Mendeley Data,
doi:10.17632/n9xttk45df.1), CC BY 4.0.
"""


def blind_dir() -> Path:
    """Drive folder handed to raters: blind clips, rating_sheet.csv, INSTRUCTIONS.md."""
    return final_dir() / "blind"


def _level_matched(wav: np.ndarray) -> np.ndarray:
    """wav scaled to BLIND_RMS_DBFS RMS, or less if its peak would pass PEAK_MAX."""
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    gain = min(10 ** (BLIND_RMS_DBFS / 20) / rms, PEAK_MAX / float(np.abs(wav).max()))
    return (wav * gain).astype(np.float32)


def make_rating_sheet(seed: int = settings.SEED) -> int:
    """Blind clips, hidden key and an empty rating sheet from final/index.csv; returns rows.

    Per prompt, A_baseline and B_boinga become <id>_clip1/_clip2 in a seeded order that puts
    each voice first for half the prompts; both are level-matched. final/key.csv says which
    is which and is never copied to blind/ or printed (only its sha256). A rating sheet that
    raters already edited is kept, and a reshuffle under it is refused.
    """
    index = pd.read_csv(final_dir() / "index.csv", dtype=str, keep_default_na=False)
    voices = index[index["set"].isin(VOICE_SETS)].set_index(["id", "set"])
    ids = list(dict.fromkeys(index["id"]))
    rng = np.random.default_rng(seed)
    b_first = rng.permutation([i < len(ids) // 2 for i in range(len(ids))])
    key, sheet = [], []
    for pid, flip in zip(ids, b_first, strict=True):
        order = ["B_boinga", "A_baseline"] if flip else ["A_baseline", "B_boinga"]
        for clip, name in zip(("clip1", "clip2"), order, strict=True):
            row = voices.loc[(pid, name)]
            key.append({"prompt_id": pid, "clip": clip, "set": name, "source": row["path"]})
            sheet.append({"prompt_id": pid, "clip": clip, "text": row["text"]})
    key_df = pd.DataFrame(key)
    sheet_df = pd.DataFrame(sheet).reindex(columns=SHEET_COLS, fill_value="")
    sheet_text = sheet_df.to_csv(index=False, lineterminator="\n")
    blind, key_csv = blind_dir(), final_dir() / "key.csv"
    sheet_csv = blind / "rating_sheet.csv"
    edited = sheet_csv.is_file() and sheet_csv.read_text(encoding="utf-8") != sheet_text
    key_text = key_df.to_csv(index=False, lineterminator="\n")
    order = ["prompt_id", "clip", "set"]  # not the source paths: a new Drive mount may move them
    same = key_csv.is_file() and pd.read_csv(key_csv, dtype=str)[order].equals(key_df[order])
    rated = edited or any(ratings_dir().glob("*.csv"))
    if rated and not same:
        raise ValueError("ratings exist (blind/rating_sheet.csv edited or ratings/*.csv) and a "
                         "different seed would change the clip order under them: move them "
                         "away before reshuffling")
    blind.mkdir(parents=True, exist_ok=True)
    for k in tqdm(key, desc="blind clips", unit="wav"):
        what = f"{k['prompt_id']} {k['clip']}"  # never the set name: errors must not unblind
        try:
            wav, rate = sf.read(k["source"], dtype="float32")
        except (OSError, RuntimeError):
            raise FileNotFoundError(f"{what}: audio missing or unreadable, run `boinga "
                                    "eval-generate` first") from None
        if not np.any(wav):
            raise ValueError(f"{what}: empty or silent audio")
        sf.write(blind / f"{k['prompt_id']}_{k['clip']}.wav", _level_matched(wav), rate,
                 subtype="PCM_16")
    key_csv.write_text(key_text, encoding="utf-8", newline="\n")
    if edited:
        print(f"WARN: kept the edited {sheet_csv} (same clip order)")
    else:
        sheet_csv.write_text(sheet_text, encoding="utf-8", newline="\n")
    (blind / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8", newline="\n")
    repo_path, _ = report_io.write_report(INSTRUCTIONS, INSTRUCTIONS_REL)
    report_io.emit(repo_path, INSTRUCTIONS_REL)
    print(f"blind clips in {blind}; key sha256 {hashlib.sha256(key_text.encode()).hexdigest()}")
    return len(sheet_df)


# --- C10.4: scoring --------------------------------------------------------------------------

CRITERIA = ["intelligibility", "naturalness", "dialect_authenticity", "dialect_words"]
EVAL_REL = "reports/eval.md"
_SCORE = re.compile(r"^[1-5]$")
LIMITATIONS = """\
## Limitations

- Informal panel of {n} rater(s), not a formal MOS test; no significance testing, so small
  differences between A and B may be noise.
- One target voice (fine-tuned on the corpus's mixed clean clips): this measures unseen
  sentences in a known voice, not new speakers.
- The base voice (tts_models/bn/custom/vits-male) was trained on Indian Bengali data; its
  accent can colour both A and B.
- The model reads graphemes (Bengali script, NFC + char map), not phonemes, so spelling
  variation in the dialect text reaches the model directly.
- ASR word error rate (WER) is not used: standard-Bangla ASR penalises correct Chittagonian forms.
"""


def ratings_dir() -> Path:
    """Drive folder of the filled rating sheets, one <rater>.csv per rater."""
    return final_dir() / "ratings"


def _read_sheets(key: pd.DataFrame) -> pd.DataFrame:
    """All rating sheets, checked against the key; raises with every bad cell listed."""
    files = sorted(ratings_dir().glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no rating sheets (*.csv) in {ratings_dir()}")
    want = set(zip(key["prompt_id"], key["clip"], strict=True))
    frames, bad = [], []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        need = ["prompt_id", "clip", *CRITERIA, "comments"]
        if missing_cols := [c for c in need if c not in df]:
            raise ValueError(f"{f.name}: missing columns {missing_cols}")
        df = df.assign(prompt_id=df["prompt_id"].str.strip(), clip=df["clip"].str.strip())
        have = list(zip(df["prompt_id"], df["clip"], strict=True))
        if dups := sorted({f"{p} {c}" for p, c in have if have.count((p, c)) > 1}):
            raise ValueError(f"{f.name}: duplicate rows {dups}")
        if extra := sorted({f"{p} {c}" for p, c in have if (p, c) not in want}):
            raise ValueError(f"{f.name}: rows {extra} are not in the key")
        if missing := sorted(f"{p} {c}" for p, c in want - set(have)):
            raise ValueError(f"{f.name}: missing rows {missing}")
        for row, r in df.iterrows():
            bad += [f"{f.name} row {int(row) + 2} {c}: {r[c]!r}" for c in CRITERIA
                    if not _SCORE.match(r[c].strip())]
        frames.append(df.assign(rater=f"R{i}"))
    if bad:
        raise ValueError(f"{len(bad)} bad cell(s); scores must be whole numbers 1-5, fix the "
                         "sheet(s):\n" + "\n".join(bad))
    out = pd.concat(frames, ignore_index=True)
    return out.astype({c: int for c in CRITERIA})


def score() -> int:
    """Unblind the rating sheets with final/key.csv and write + emit reports/eval.md.

    Returns the number of raters. Raters appear as R1..Rn (sheets sorted by file name).
    """
    key = pd.read_csv(final_dir() / "key.csv", dtype=str)
    df = _read_sheets(key).merge(key[["prompt_id", "clip", "set"]], on=["prompt_id", "clip"])
    df["overall"] = df[CRITERIA].mean(axis=1)
    raters = sorted(df["rater"].unique(), key=lambda r: int(r[1:]))
    lines = ["# Blind listening test", "",
             (f"{len(raters)} raters ({', '.join(raters)}), {df['prompt_id'].nunique()} prompts, "
             f"{len(df)} ratings. A_baseline = pretrained Bangla voice, B_boinga = fine-tuned "
             "voice (Drive best/). Scores 1-5; mean ± sd over all ratings."), "",
             "| criterion | A_baseline | B_boinga | B − A |", "|---|---|---|---|"]
    for c in [*CRITERIA, "overall"]:
        a, b = (df.loc[df["set"] == s, c] for s in VOICE_SETS)
        lines.append(f"| {c} | {a.mean():.2f} ± {a.std():.2f} | {b.mean():.2f} ± "
                     f"{b.std():.2f} | {b.mean() - a.mean():+.2f} |")
    wide = df.pivot_table(index=["prompt_id", "rater"], columns="set", values="overall")
    wide["win"] = np.sign(wide["B_boinga"] - wide["A_baseline"]) / 2 + 0.5
    per = wide.groupby(level="prompt_id").mean()
    lines += ["", "## Per prompt", "",
              ("Mean of the 4 criteria; B win rate = share of raters scoring B above A (tie = "
              "0.5)."), "", "| prompt | A_baseline | B_boinga | B win rate |", "|---|---|---|---|"]
    lines += [f"| {p} | {r.A_baseline:.2f} | {r.B_boinga:.2f} | {r.win:.2f} |"
              for p, r in per.iterrows()]
    lines += ["", f"B_boinga win rate over prompts: {per['win'].mean():.2f}", "", "## Comments"]
    for s in VOICE_SETS:
        notes = df[(df["set"] == s) & (df["comments"].str.strip() != "")]
        lines += ["", f"### {s}", ""]
        lines += [f"- {r.prompt_id} ({r.rater}): {' '.join(r.comments.split())}"
                  for r in notes.itertuples()] or ["(none)"]
    text = "\n".join(lines) + "\n\n" + LIMITATIONS.format(n=len(raters))
    repo_path, _ = report_io.write_report(text, EVAL_REL)
    report_io.emit(repo_path, EVAL_REL)
    return len(raters)
