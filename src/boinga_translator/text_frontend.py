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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from boinga_translator import paths

CHAR_MAP_FILE = "char_map.json"
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
