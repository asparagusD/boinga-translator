"""Tests for text_frontend.check_tokenizer and `boinga check-tokenizer` with a fake coqui tokenizer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from boinga_translator import cli, paths, pretrained, text_frontend, text_oov

REAL_MAP = paths.configs_dir() / text_frontend.CHAR_MAP_FILE  # resolved before any monkeypatch


class FakeCharacters:
    """coqui Graphemes stand-in: special tokens first, then the vocabulary."""

    def __init__(self, vocab: str, lossy: dict[str, str] | None = None) -> None:
        self.pad_id, self.eos_id, self.bos_id, self.blank_id = 0, 1, 2, 3
        self._vocab = ["<PAD>", "<EOS>", "<BOS>", "<BLNK>", *vocab]
        self._lossy = lossy or {}  # id_to_char returns another character (a broken vocab)

    def char_to_id(self, char: str) -> int:
        try:
            return self._vocab.index(char, 4)
        except ValueError:
            raise KeyError(char) from None

    def id_to_char(self, idx: int) -> str:
        char = self._vocab[idx]
        return self._lossy.get(char, char)


class FakeTokenizer:
    """Mirrors coqui 0.28 TTSTokenizer: encode drops unknown chars, blanks, bos/eos padding."""

    def __init__(self, vocab: str, add_blank: bool = True, use_eos_bos: bool = False,
                 lossy: dict[str, str] | None = None) -> None:
        self.characters = FakeCharacters(vocab, lossy)
        self.add_blank, self.use_eos_bos = add_blank, use_eos_bos
        self.text_cleaner, self.use_phonemes = None, False
        self.not_found_characters: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            try:
                ids.append(self.characters.char_to_id(char))
            except KeyError:
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.characters.id_to_char(i) for i in ids)

    def text_to_ids(self, text: str) -> list[int]:
        ids = self.encode(text)
        if self.add_blank:
            out = [self.characters.blank_id] * (len(ids) * 2 + 1)
            out[1::2] = ids
            ids = out
        if self.use_eos_bos:
            ids = [self.characters.bos_id, *ids, self.characters.eos_id]
        return ids


VOCAB = "আইযামক .,"  # আ ই য া ম ক, space, . ,


@pytest.mark.parametrize(("add_blank", "use_eos_bos"), [(True, False), (False, True),
                                                        (True, True), (False, False)])
def test_round_trip_ok_whatever_the_padding(add_blank: bool, use_eos_bos: bool) -> None:
    tok = FakeTokenizer(VOCAB, add_blank=add_blank, use_eos_bos=use_eos_bos)
    rt = text_frontend.round_trip("7", " আই  যাম।", tok)
    assert rt.text == "আই যাম."  # prepare_text: danda -> "."
    assert rt.decoded == rt.text and rt.unknown == "" and rt.ok


def test_round_trip_reports_discarded_and_lossy_characters() -> None:
    rt = text_frontend.round_trip("8", "আখইখ", FakeTokenizer(VOCAB))
    assert rt.unknown == "খখ" and rt.decoded == "আই" and not rt.ok
    lossy = FakeTokenizer(VOCAB, lossy={"ক": "আ"})
    rt = text_frontend.round_trip("9", "কই", lossy)
    assert rt.unknown == "" and rt.decoded == "আই" and not rt.ok  # vocab mismatch


def test_render_rates_tables_and_full_failures() -> None:
    tok = FakeTokenizer(VOCAB)
    texts = {"1": "আই", "2": "আখ", "3": "খগ ই",
             **{f"x{i}": f"খ{i}" for i in range(12)}}  # digits too: not in VOCAB
    results = [text_frontend.round_trip(cid, t, tok) for cid, t in texts.items()]
    md = text_frontend.render_tokenizer_check(results, Path("cfg/config.json"))
    n_chars = sum(len(r.text) for r in results)
    n_unknown = sum(len(r.unknown) for r in results)
    assert f"- Unknown character rate: {n_unknown} of {n_chars} characters" in md
    assert "- Round-trip failures: 14 of 15 clips" in md
    assert "| 3 | U+0996 U+0997 | no |" in md  # unknown characters as code points
    assert "| 2 | U+0996 | no |" in md
    assert md.count("~~~text") == 2 * text_frontend.N_FULL_FAILURES  # first 10 in full
    full = md.split("## First")[1]
    assert "clip x7:" in full and "clip x8:" not in full  # the 10th failure is x7, 11th x8
    assert "| x0 | U+0996 U+0030 | no |" in md
    rows = [text_frontend.round_trip("r", "খখগখ", tok)]
    assert "| r | U+0996 U+0997 | no |" in text_frontend.render_tokenizer_check(
        rows, Path("c.json"))  # distinct characters only
    assert "cfg/config.json" in md


def test_render_when_everything_round_trips() -> None:
    results = [text_frontend.round_trip("1", "আই", FakeTokenizer(VOCAB))]
    md = text_frontend.render_tokenizer_check(results, Path("c.json"))
    assert "0 of 2 characters (0.000000)" in md and "0 of 1 clips" in md
    assert "Every clip round-trips" in md


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Fake repo + Drive that still holds the real char map."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    paths.configs_dir().mkdir(parents=True)
    (paths.configs_dir() / text_frontend.CHAR_MAP_FILE).write_bytes(REAL_MAP.read_bytes())
    text_frontend.default_char_map.cache_clear()
    yield tmp_path
    text_frontend.default_char_map.cache_clear()  # don't leak the fake repo's map


def test_check_tokenizer_writes_emits_and_passes(fake_repo: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    rate = text_frontend.check_tokenizer({"1": "আই।", "2": "আখ"},
                                         Path("c.json"), tokenizer=FakeTokenizer(VOCAB))
    assert rate == pytest.approx(1 / 5)
    out = capsys.readouterr().out
    assert "=== BEGIN reports/tokenizer_check.md ===" in out
    assert "PASS: tokenizer check unknown_rate=0.200000 round_trip_failures=1" in out
    assert (paths.REPO / text_frontend.TOKENIZER_REL).is_file()


def test_check_tokenizer_refuses_cleaner_or_phonemes(fake_repo: Path) -> None:
    tok = FakeTokenizer(VOCAB)
    tok.use_phonemes = True
    with pytest.raises(ValueError, match="phonemes"):
        text_frontend.check_tokenizer({"1": "আ"}, Path("c.json"), tokenizer=tok)


def test_cli_check_tokenizer(fake_repo: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(text_oov, "selected_texts",
                        lambda: ({"1": "আই।"}, 2, "male"))
    monkeypatch.setattr(pretrained, "fetch",
                        lambda voice: {"config": Path(f"/drive/{voice}/config.json")})
    seen: list[Path] = []
    monkeypatch.setattr(text_frontend, "load_tokenizer",
                        lambda path: seen.append(path) or FakeTokenizer(VOCAB))
    assert cli.main(["check-tokenizer"]) == 0
    assert seen == [Path("/drive/male/config.json")]  # the base voice's config
    out = capsys.readouterr().out
    assert "voice male: 1 clips, 2 excluded by char_map" in out
    assert "PASS: tokenizer check unknown_rate=0.000000 round_trip_failures=0" in out


def test_cli_check_tokenizer_fails_cleanly(fake_repo: Path, monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    def no_coqui(path: Path) -> FakeTokenizer:
        raise ImportError("No module named 'TTS'")

    monkeypatch.setattr(text_oov, "selected_texts", lambda: ({"1": "আ"}, 0, "male"))
    monkeypatch.setattr(pretrained, "fetch", lambda voice: {"config": Path("c.json")})
    monkeypatch.setattr(text_frontend, "load_tokenizer", no_coqui)
    assert cli.main(["check-tokenizer"]) == 1
    assert "FAIL: check-tokenizer: ImportError: No module named 'TTS'" in capsys.readouterr().out
