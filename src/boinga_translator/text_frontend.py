"""Text front-end: NFC + configs/char_map.json for every text the model sees (train and infer)."""

# DECISIONS "Character map": prepare() = corpus.normalise_text (NFC, collapse, strip), then
# the ordered char_map replacements, then collapse/strip again. Its output is deliberately
# not NFC (precomposed nukta letters), so nothing may re-normalise it afterwards; prepare()
# itself is idempotent because the map recomposes what NFC decomposes. Rows containing an
# exclude_rows_with character are dropped from training, never rewritten.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from boinga_translator import corpus, paths

CHAR_MAP_FILE = "char_map.json"


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


def prepare(text: str, cmap: CharMap | None = None) -> str:
    """The text the model reads: NFC, collapse/strip, char_map, collapse/strip again."""
    cmap = cmap or default_char_map()
    out = corpus.normalise_text(text)
    for src, dst in cmap.pairs:
        out = out.replace(src, dst)
    return " ".join(out.split())


def excluded(text: str, cmap: CharMap | None = None) -> bool:
    """True if the row contains a character that excludes it from training (e.g. digits)."""
    cmap = cmap or default_char_map()
    return any(c in cmap.exclude for c in corpus.normalise_text(text))
