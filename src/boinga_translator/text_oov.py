"""OOV report: characters in the selected clips' model text that the base voice's charset lacks."""

# The texts are what the model will read: text_frontend.prepare(ctg_text_nfc) of the C4
# speaker selection, without the rows char_map excludes (they never reach training). The
# charset is the base voice's (configs/speaker_selection.json), via data_audit. Report only:
# nothing is mapped or dropped here (dialect-data-rules); Gate G4 (C5.5) judges the numbers.

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd

from boinga_translator import corpus, data_audit, report_io, speaker_select, text_frontend

OOV_REL = "reports/oov_report.md"
OOV_COLS = ("char", "codepoint", "name", "count", "clips_affected", "examples")
N_EXAMPLES = 3


def oov_report(texts: Iterable[str], charset: frozenset[str] | set[str]) -> pd.DataFrame:
    """One row per character not in charset: code point, name, count, clips, 3 example words.

    Whitespace is never reported. Rows are sorted by count (high first), then code point.
    """
    count: Counter[str] = Counter()
    clips: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for text in texts:
        seen: set[str] = set()
        for word in text.split():
            for c in word:
                if c in charset:
                    continue
                count[c] += 1
                seen.add(c)
                words = examples.setdefault(c, [])
                if word not in words and len(words) < N_EXAMPLES:
                    words.append(word)
        clips.update(seen)
    rows = [(c, f"U+{ord(c):04X}", unicodedata.name(c, "(unnamed)"), n, clips[c], examples[c])
            for c, n in sorted(count.items(), key=lambda kv: (-kv[1], ord(kv[0])))]
    return pd.DataFrame(rows, columns=list(OOV_COLS))


def clips_with_oov(texts: Iterable[str], charset: frozenset[str] | set[str]) -> int:
    """How many texts contain at least one non-space character outside charset."""
    return sum(any(c not in charset and not c.isspace() for c in t) for t in texts)


_MD_SPECIAL = "\\`*_|"


def _cell(text: str) -> str:
    """Plain-text table cell: backslash-escape Markdown specials and the column bar."""
    return "".join(f"\\{c}" if c in _MD_SPECIAL else c for c in text)


def _rate(part: int, whole: int, unit: str) -> str:
    return f"{part} of {whole} {unit} ({100 * part / whole if whole else 0.0:.2f}%)"


def render(texts: Sequence[str], charset: frozenset[str] | set[str], voice: str,
           n_excluded: int, df: pd.DataFrame | None = None) -> str:
    """reports/oov_report.md: overall OOV character rate, share of clips, per-character table.

    Pass df (oov_report of the same texts and charset) to avoid counting twice.
    """
    df = oov_report(texts, charset) if df is None else df
    n_chars = sum(len(t) - sum(c.isspace() for c in t) for t in texts)
    n_hit = clips_with_oov(texts, charset)
    lines = [
        "# OOV report", "",
        (f"Characters of the model text (text_frontend.prepare, char map "
        f"{text_frontend.map_digest()}) missing from the base voice **{voice}**'s charset "
        f"(characters | punctuations | space). Made by `boinga oov`."), "",
        (f"- Clips: {len(texts)} selected clips checked; {n_excluded} selected clips excluded "
        "by char_map exclude_rows_with (not checked: they never reach training)"),
        f"- OOV character rate: {_rate(int(df['count'].sum()), n_chars, 'characters')}",
        f"- Clips with any OOV character: {_rate(n_hit, len(texts), 'clips')}", ""]
    if df.empty:
        lines.append("No character is missing from the charset.")
    else:
        lines += ["| char | code point | name | count | clips | examples |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {_cell(r.char)} | {r.codepoint} | {r.name} | {r.count} | "
                  f"{r.clips_affected} | {_cell(', '.join(r.examples))} |"
                  for r in df.itertuples()]
    return "\n".join(lines) + "\n"


def run() -> pd.DataFrame:
    """`boinga oov`: report the selected speaker's model text against the base voice."""
    selection, sel = speaker_select.load_selection()
    voice = str(sel["base_voice"])
    index = corpus.load_index().set_index("clip_id")
    if missing := sorted(set(selection["clip_id"]) - set(index.index)):
        raise ValueError(f"{len(missing)} selected clip(s) not in the corpus index: "
                         f"{missing[:5]}")
    raw = index.loc[selection["clip_id"], "ctg_text_nfc"]
    kept = [text_frontend.prepare(t) for t in raw if not text_frontend.excluded(t)]
    charset = data_audit.load_model_charset(voice)
    df = oov_report(kept, charset)
    repo_path, _ = report_io.write_report(
        render(kept, charset, voice, n_excluded=len(raw) - len(kept), df=df), OOV_REL)
    report_io.emit(repo_path, OOV_REL)
    n_hit = clips_with_oov(kept, charset)
    print(f"PASS: oov report: {len(df)} missing character(s) in {n_hit} of {len(kept)} clips")
    return df
