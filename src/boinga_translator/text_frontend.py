"""Text front-end: NFC-only normalise() plus configs/char_map.json for every text the model sees."""

# DECISIONS "Character map": prepare_text() (= prepare()) = normalise() (NFC, collapse,
# strip), then apply_char_map() (the ordered replacements), then collapse/strip again. Its
# output is deliberately not NFC (precomposed nukta letters), so nothing may re-normalise it
# afterwards; it is idempotent because the map recomposes what NFC decomposes. Rows containing an
# exclude_rows_with character are dropped from training, never rewritten.

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from boinga_translator import paths, report_io

CHAR_MAP_FILE = "char_map.json"
TOKENIZER_REL = "reports/tokenizer_check.md"
N_FULL_FAILURES = 10  # failures shown with their full text and decoded text
N_LISTED_FAILURES = 200  # cap on the failing-clip table (the report must stay emit-sized)
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """The only default normalisation: Unicode NFC, collapse whitespace runs, strip.

    Nothing else (dialect-data-rules): no character mapping, no spelling changes. NFC itself
    decomposes the composition-excluded nukta letters (U+09DC -> U+09A1 U+09BC, likewise
    U+09DD, U+09DF) and joins two-part vowel signs (U+09C7 U+09BE -> U+09CB).
    ZWJ/ZWNJ are kept; NBSP and other Unicode whitespace count as whitespace.
    """
    return _SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


@dataclass(frozen=True)
class CharMap:
    """Ordered (from, to) replacements and the characters that exclude a row."""

    pairs: tuple[tuple[str, str], ...]
    exclude: frozenset[str]


def load_char_map(path: Path | None = None) -> CharMap:
    """Read and validate a char map file (default configs/char_map.json)."""
    raw = json.loads((path or paths.configs_dir() / CHAR_MAP_FILE).read_text(encoding="utf-8"))
    pairs = []
    for entry in raw["map"]:
        if not entry.get("from"):
            raise ValueError(f"char_map entry with an empty 'from': {entry}")
        pairs.append((entry["from"], entry["to"]))
    return CharMap(tuple(pairs), frozenset(raw["exclude_rows_with"]["chars"]))


@lru_cache(maxsize=1)
def default_char_map() -> CharMap:
    """configs/char_map.json, read once per process."""
    return load_char_map()


def map_digest(cmap: CharMap | None = None) -> str:
    """Short hash of a char map, so audit rows made with another map count as stale.

    Hashes the parsed pairs and exclusions, not the file: editing a note changes nothing.
    """
    cmap = cmap or default_char_map()
    blob = json.dumps([cmap.pairs, sorted(cmap.exclude)], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def apply_char_map(text: str, mapping: CharMap | None = None) -> str:
    """Only the ordered char_map replacements (default configs/char_map.json), nothing else.

    Characters not in the map, and whitespace, are left exactly as they are.
    """
    cmap = default_char_map() if mapping is None else mapping
    for src, dst in cmap.pairs:
        text = text.replace(src, dst)
    return text


def prepare_text(text: str, mapping: CharMap | None = None) -> str:
    """The text the model reads: apply_char_map(normalise(text)), whitespace re-collapsed.

    Removing a character (quote, bracket) can leave a double or edge space, so whitespace
    is collapsed and stripped once more. Not NFC-stable: never re-normalise the output.
    """
    return " ".join(apply_char_map(normalise(text), mapping).split())


def prepare(text: str, cmap: CharMap | None = None) -> str:
    """Same as prepare_text (the name the audit, baseline and OOV report use)."""
    return prepare_text(text, cmap)


def count_unknown(text: str, charset: frozenset[str] | set[str]) -> int:
    """Occurrences (not distinct characters) of non-whitespace characters not in charset."""
    return sum(1 for c in text if c not in charset and not c.isspace())


def excluded(text: str, cmap: CharMap | None = None) -> bool:
    """True if the row contains a character that excludes it from training (e.g. digits)."""
    cmap = cmap or default_char_map()
    return any(c in cmap.exclude for c in normalise(text))


# --- coqui tokenizer round trip (C5.4) -----------------------------------------------------
# coqui-tts 0.28 TTSTokenizer.text_to_ids: cleaner -> encode (an unknown character is
# dropped with a log warning) -> blanks between tokens (add_blank) -> bos/eos (use_eos_bos).
# The pretrained configs have no cleaner and no phonemes (configs/pretrained_facts.json),
# so the text the tokenizer sees is prepare_text() output, and decoding the non-blank,
# non-bos/eos ids must give it back exactly.


def load_tokenizer(config_path: Path) -> Any:
    """coqui's TTSTokenizer for a voice's config.json (coqui is imported only here)."""
    from TTS.config import load_config
    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    tokenizer, _ = TTSTokenizer.init_from_config(load_config(str(config_path)))
    return tokenizer


@dataclass(frozen=True)
class RoundTrip:
    """One clip through the tokenizer: its model text, the decoded ids, dropped characters."""

    clip_id: str
    text: str  # prepare_text() output, what text_to_ids receives
    decoded: str  # decode() of the ids without blanks and bos/eos
    unknown: str  # characters encode() discards, in text order

    @property
    def ok(self) -> bool:
        """Nothing dropped and decode gives the model text back exactly."""
        return not self.unknown and self.decoded == self.text


def _known(tokenizer: Any, char: str) -> bool:
    try:
        tokenizer.characters.char_to_id(char)
    except KeyError:
        return False
    return True


def round_trip(clip_id: str, raw_text: str, tokenizer: Any) -> RoundTrip:
    """prepare_text(raw_text) -> text_to_ids -> strip bos/eos and blanks -> decode."""
    text = prepare_text(raw_text)
    ids = list(tokenizer.text_to_ids(text))
    if tokenizer.use_eos_bos:
        ids = ids[1:-1]
    if tokenizer.add_blank:
        ids = ids[1::2]  # blanks sit at the even positions
    unknown = "".join(c for c in text if not _known(tokenizer, c))
    return RoundTrip(clip_id, text, tokenizer.decode(ids), unknown)


def _codepoints(text: str) -> str:
    """Distinct characters as code points, first seen first (keeps table rows short)."""
    return " ".join(f"U+{ord(c):04X}" for c in dict.fromkeys(text))


def unknown_rate(results: Sequence[RoundTrip]) -> tuple[int, int, float]:
    """(unknown characters, characters, their ratio) over all clips' model text."""
    n_chars = sum(len(r.text) for r in results)
    n_unknown = sum(len(r.unknown) for r in results)
    return n_unknown, n_chars, n_unknown / n_chars if n_chars else 0.0


def render_tokenizer_check(results: Sequence[RoundTrip], config_path: Path) -> str:
    """reports/tokenizer_check.md: unknown-char rate, failing clips, first failures in full."""
    n_unknown, n_chars, rate = unknown_rate(results)
    failures = [r for r in results if not r.ok]
    lines = [
        "# Tokenizer check", "",
        (f"coqui TTSTokenizer from `{config_path.as_posix()}` on prepare_text() of each clip "
        f"(char map {map_digest()}). Made by `boinga check-tokenizer`."), "",
        f"- Clips checked: {len(results)}",
        f"- Unknown character rate: {n_unknown} of {n_chars} characters ({rate:.6f})",
        f"- Round-trip failures: {len(failures)} of {len(results)} clips", ""]
    if not failures:
        lines.append("Every clip round-trips: decode(encode(text)) == text, nothing dropped.")
        return "\n".join(lines) + "\n"
    lines += ["## Failing clips", "", "| clip | unknown characters | decoded == text |",
              "|---|---|---|"]
    lines += [f"| {r.clip_id} | {_codepoints(r.unknown)} | {'yes' if r.decoded == r.text else 'no'} |"
              for r in failures[:N_LISTED_FAILURES]]
    if len(failures) > N_LISTED_FAILURES:
        lines.append(f"\n... and {len(failures) - N_LISTED_FAILURES} more failing clips")
    lines += ["", f"## First {N_FULL_FAILURES} failures in full", ""]
    for r in failures[:N_FULL_FAILURES]:
        lines += [f"clip {r.clip_id}: text, then decoded", "", "~~~text", r.text, "~~~",
                  "~~~text", r.decoded, "~~~", ""]
    return "\n".join(lines) + "\n"


def check_tokenizer(texts: Mapping[str, str], config_path: Path,
                    tokenizer: Any | None = None) -> float:
    """Round-trip every clip's text (clip id -> raw text) through the voice's tokenizer.

    Writes and emits reports/tokenizer_check.md and returns the unknown-character rate.
    """
    tok = load_tokenizer(config_path) if tokenizer is None else tokenizer
    if tok.text_cleaner is not None or tok.use_phonemes:
        raise ValueError("tokenizer uses a text cleaner or phonemes; this check assumes "
                         "neither (configs/pretrained_facts.json)")
    results = [round_trip(cid, text, tok) for cid, text in texts.items()]
    repo_path, _ = report_io.write_report(render_tokenizer_check(results, config_path),
                                          TOKENIZER_REL)
    report_io.emit(repo_path, TOKENIZER_REL)
    rate = unknown_rate(results)[2]
    n_fail = sum(not r.ok for r in results)
    print(f"PASS: tokenizer check unknown_rate={rate:.6f} round_trip_failures={n_fail}")
    return rate
