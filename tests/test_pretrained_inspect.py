"""Tests for pretrained.inspect(): config facts, checkpoint facts, charset table, reports, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boinga_translator import cli, paths, pretrained

CFG = {
    "model": "vits",
    "audio": {"sample_rate": 22050, "hop_length": 256},
    "characters": {"characters_class": "TTS.tts.models.vits.VitsCharacters",
                   "characters": "কখ ক", "punctuations": "!।", "pad": "<PAD>",
                   "eos": "<EOS>", "bos": "<BOS>", "blank": "<BLNK>", "phonemes": None},
    "use_phonemes": False, "text_cleaner": "basic_cleaners", "add_blank": True,
    "model_args": {"num_speakers": 0, "use_speaker_embedding": False},
    "output_path": "/tmp/boinga-female-x/run",
}
CKPT = {"model": {"disc.a.weight": np.zeros((2, 3)), "text_encoder.emb.weight": np.zeros((4, 5)),
                  "dec.bias": np.zeros(4)},
        "step": 7, "config": {}}


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """REPO and ROOT under tmp_path; both voices cached with CFG; checkpoint loader faked."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    for voice in pretrained.voices():
        d = paths.pretrained() / voice
        d.mkdir(parents=True)
        (d / "model.pth").write_bytes(b"w")
        (d / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
    monkeypatch.setattr(pretrained, "_load_checkpoint", lambda path: CKPT)
    return repo


def test_config_facts_picks_keys_and_flags_absolute_paths() -> None:
    f = pretrained.config_facts(CFG)
    assert f["audio"]["sample_rate"] == 22050
    assert f["characters"]["blank"] == "<BLNK>"
    assert f["text_cleaner"] == "basic_cleaners" and f["add_blank"] is True
    assert f["phonemizer"] is None and f["enable_eos_bos_chars"] is None  # missing -> None
    assert f["num_speakers"] == 0 and f["model"] == "vits"
    assert f["absolute_paths"] == {"output_path": "/tmp/boinga-female-x/run"}


def test_absolute_paths_walks_lists_skips_characters_and_sees_windows() -> None:
    cfg = {"datasets": [{"path": "/tmp/boinga-male-y/data"}],
           "stats_path": "C:\\tmp\\stats.npy", "characters": {"punctuations": "/!"},
           "relative": "a/b"}
    assert pretrained.config_facts(cfg)["absolute_paths"] == {
        "datasets.0.path": "/tmp/boinga-male-y/data", "stats_path": "C:\\tmp\\stats.npy"}


def test_config_facts_without_characters_block() -> None:
    f = pretrained.config_facts({"audio": {"sample_rate": 16000}})
    assert f["characters"]["characters"] is None
    assert pretrained.charset_lines(f["characters"]) == ""


def test_checkpoint_facts_counts_disc_params() -> None:
    f = pretrained.checkpoint_facts(CKPT)
    assert f["top_level_keys"] == ["config", "model", "step"]
    assert f["n_params"] == 6 + 20 + 4
    assert (f["n_disc_keys"], f["n_disc_params"], f["has_discriminator"]) == (1, 6, True)
    assert f["step"] == 7


def test_checkpoint_facts_bare_state_dict_without_disc() -> None:
    f = pretrained.checkpoint_facts({"dec.w": np.zeros((3, 3))})
    assert (f["n_params"], f["has_discriminator"], f["step"]) == (9, False, None)


def test_charset_lines_dedupes_and_names() -> None:
    lines = pretrained.charset_lines(CFG["characters"]).splitlines()
    assert lines[0] == "ক\tU+0995\tBENGALI LETTER KA"
    assert lines[2] == "\\u0020\tU+0020\tSPACE"  # whitespace shown escaped
    assert [ln.split("\t")[1] for ln in lines] == ["U+0995", "U+0996", "U+0020", "U+0021",
                                                     "U+0964"]


def test_inspect_writes_and_emits_reports(
        sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    facts_path = sandbox / "configs" / "pretrained_facts.json"
    facts_path.parent.mkdir()
    facts_path.write_text(json.dumps({"old": {"x": 1}}), encoding="utf-8")
    results = {"female": pretrained.inspect("female")}
    pretrained.write_inspect_reports(results)
    out = capsys.readouterr().out
    assert "PASS: inspected female" in out
    for rel in ("reports/pretrained_charset_female.txt", "configs/pretrained_facts.json",
                "reports/pretrained_inspect.md"):
        assert out.count(f"=== BEGIN {rel} ===") == 1
        assert (sandbox / rel).is_file()
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert facts["old"] == {"x": 1}  # other entries kept
    assert facts["female"] == {
        "sample_rate": 22050, "characters": "কখ ক", "punctuations": "!।", "pad": "<PAD>",
        "eos": "<EOS>", "bos": "<BOS>", "blank": "<BLNK>",
        "characters_class": "TTS.tts.models.vits.VitsCharacters", "use_phonemes": False,
        "text_cleaner": "basic_cleaners", "add_blank": True, "has_discriminator": True,
        "n_params": 30, "n_disc_params": 6}
    md = (sandbox / "reports" / "pretrained_inspect.md").read_text(encoding="utf-8")
    assert "## female" in md and "/tmp/boinga-female-x/run" in md


def test_inspect_needs_cached_voice(sandbox: Path) -> None:
    (paths.pretrained() / "male" / "config.json").unlink()
    with pytest.raises(RuntimeError, match="fetch-pretrained"):
        pretrained.inspect("male")


def test_inspect_unknown_voice(sandbox: Path) -> None:
    with pytest.raises(ValueError, match="unknown voice"):
        pretrained.inspect("robot")


def test_load_checkpoint_with_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    p = tmp_path / "model.pth"
    torch.save({"model": {"disc.w": torch.zeros(2, 2)}, "step": 3}, p)
    f = pretrained.checkpoint_facts(pretrained._load_checkpoint(p))
    assert (f["n_disc_params"], f["step"]) == (4, 3)


def test_cli_inspect_both_writes_reports_once(
        sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["inspect-pretrained", "--voice", "both"]) == 0
    out = capsys.readouterr().out
    assert "PASS: inspected female" in out and "PASS: inspected male" in out
    assert out.count("=== BEGIN configs/pretrained_facts.json ===") == 1
    assert set(json.loads((sandbox / "configs" / "pretrained_facts.json").read_text(
        encoding="utf-8"))) == {"female", "male"}


def test_cli_inspect_failure_exits_1(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (paths.pretrained() / "male" / "model.pth").unlink()
    assert cli.main(["inspect-pretrained", "--voice", "both"]) == 1
    out = capsys.readouterr().out
    assert "PASS: inspected female" in out and "FAIL: male:" in out


def test_cli_inspect_reports_load_errors_and_bad_facts(
        sandbox: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    def bad_load(path: Path) -> None:
        raise ImportError("no torch")
    monkeypatch.setattr(pretrained, "_load_checkpoint", bad_load)
    assert cli.main(["inspect-pretrained", "--voice", "female"]) == 1
    assert "FAIL: female: ImportError: no torch" in capsys.readouterr().out
    monkeypatch.setattr(pretrained, "_load_checkpoint", lambda path: CKPT)
    (sandbox / "configs").mkdir()
    (sandbox / "configs" / "pretrained_facts.json").write_text("{oops", encoding="utf-8")
    assert cli.main(["inspect-pretrained", "--voice", "female"]) == 1
    assert "FAIL: reports: configs/pretrained_facts.json is not valid JSON" in \
        capsys.readouterr().out
