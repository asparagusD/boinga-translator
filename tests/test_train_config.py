"""Tests for train_config.py and `boinga build-config`: overrides, protected keys, outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from boinga_translator import cli, paths, settings, train_config

SR = 22050
FACTS = {"dataset": {"formatter": "ljspeech", "meta_files": {
    "train": "metadata_train.csv", "val": "metadata_val.csv", "test": "metadata_test.csv"}}}
PRETRAINED = {
    "run_name": "vits_male", "output_path": "/home/ansary/Shabab/",
    "restore_path": "/home/ansary/Shabab/vits/checkpoint_626000.pth",
    "lr_gen": 0.0002, "lr_disc": 0.0004, "batch_size": 48, "epochs": 10000, "training_seed": 54321,
    "mixed_precision": False, "test_sentences": [["আমি"]],
    "audio": {"sample_rate": SR, "fft_size": 1024, "hop_length": 256, "num_mels": 80},
    "characters": {"characters_class": "TTS.tts.utils.text.characters.Graphemes",
                   "characters": "অআক", "punctuations": "!,.? ", "pad": "<PAD>",
                   "eos": "।", "bos": "<BOS>", "blank": "<BLNK>", "is_unique": True,
                   "is_sorted": True},
    "use_phonemes": False, "text_cleaner": None, "add_blank": True,
    "model_args": {"hidden_channels": 192, "num_chars": 70},
    "datasets": [{"formatter": "ljspeech", "dataset_name": "", "path": "/home/x/mono/",
                  "meta_file_train": "/home/x/mono/metadata_male.txt", "meta_file_val": "",
                  "ignored_speakers": None, "language": "", "phonemizer": "",
                  "meta_file_attn_mask": ""}],
}
VAL = [f"v{i}|কথা {i}|কথা {i}" for i in range(7)]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake repo (selection, facts, DECISIONS), Drive cache (male config) and val manifest."""
    repo = tmp_path / "repo"
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    (repo / "configs").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "configs" / "speaker_selection.json").write_text(
        json.dumps({"base_voice": "male"}), encoding="utf-8")
    (repo / "configs" / "pretrained_facts.json").write_text(json.dumps(FACTS), encoding="utf-8")
    (repo / "docs" / "DECISIONS.md").write_text(
        "# Decisions\n\n## Other\nkeep me\n\n" + train_config.DECISIONS_BEGIN + "\nold\n"
        + train_config.DECISIONS_END + "\n\n## After\nkeep me too\n", encoding="utf-8")
    voice = paths.pretrained() / "male"
    voice.mkdir(parents=True)
    (voice / "config.json").write_text(json.dumps(PRETRAINED), encoding="utf-8")
    paths.data_prepared().mkdir(parents=True)
    (paths.data_prepared() / "metadata_val.csv").write_text("\n".join(VAL) + "\n",
                                                            encoding="utf-8")
    return repo


def _written(repo: Path) -> dict:
    return json.loads((repo / train_config.CONFIG_REL).read_text(encoding="utf-8"))


def test_base_dir_prefers_extended(repo: Path) -> None:
    assert train_config.base_dir() == paths.pretrained() / "male"
    (paths.pretrained() / "male_extended").mkdir()
    assert train_config.base_dir() == paths.pretrained() / "male_extended"


def test_build_overrides_and_protected(repo: Path) -> None:
    cfg = train_config.build("smoke")
    for key in train_config.PROTECTED:
        assert cfg[key] == PRETRAINED[key]
    assert cfg["lr_gen"] == pytest.approx(0.0001) and cfg["lr_disc"] == pytest.approx(0.0002)
    assert cfg["batch_size"] == 16 and cfg["eval_batch_size"] == 8
    assert cfg["mixed_precision"] is True and cfg["epochs"] == 1000
    assert cfg["training_seed"] == settings.SEED and cfg["run_name"] == "smoke"
    assert "seed" not in cfg  # not a coqui field: VitsConfig would drop it (C7 run 1)
    assert cfg["max_audio_len"] == 15 * SR and cfg["min_audio_len"] == SR // 2
    assert cfg["output_path"] == str(paths.runs()) and cfg["restore_path"] is None
    (ds,) = cfg["datasets"]
    assert ds["path"] == str(paths.data_prepared()) and ds["formatter"] == "ljspeech"
    assert ds["meta_file_train"] == "metadata_train.csv"
    assert ds["meta_file_val"] == "metadata_val.csv" and ds["language"] == "bn"
    assert cfg["test_sentences"] == [[f"কথা {i}"] for i in range(5)]
    assert cfg == _written(repo)
    assert PRETRAINED["lr_gen"] == 0.0002  # the fixture dict itself is untouched


def test_smoke_run_logs_every_5_steps(repo: Path) -> None:
    assert train_config.build("smoke")["print_step"] == 5
    text = (repo / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "| `print_step` | (missing) | 5 | smoke run only" in text
    assert train_config.build("full")["print_step"] == 25
    text = (repo / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "smoke run only" not in text  # the per-run reason does not leak to other runs
    assert set(train_config.RUN_OVERRIDES["smoke"]) <= set(train_config.OVERRIDES)


def test_unknown_run_override_refused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(train_config.RUN_OVERRIDES, "typo", {"print_stpe": (5, "x")})
    with pytest.raises(ValueError, match="print_stpe"):
        train_config.build("typo")


def test_every_override_has_a_reason_and_no_protected_key(repo: Path) -> None:
    cfg = train_config.build("smoke")
    keys = set(train_config.overrides(copy.deepcopy(PRETRAINED), "smoke", ["x"] * 5))
    assert keys == set(train_config.REASONS)
    assert not keys & set(train_config.PROTECTED)
    assert not train_config.author_paths(cfg)  # no /home/ansary/... left


def test_protected_change_is_refused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = train_config.overrides

    def bad(*args: object, **kw: object) -> dict:
        return {**real(*args, **kw), "add_blank": False}

    monkeypatch.setattr(train_config, "overrides", bad)
    with pytest.raises(RuntimeError, match="add_blank"):
        train_config.build("smoke")
    assert not (repo / train_config.CONFIG_REL).exists()


def test_missing_lr_is_refused(repo: Path) -> None:
    cfg = dict(PRETRAINED)
    del cfg["lr_disc"]
    (paths.pretrained() / "male" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="lr_disc"):
        train_config.build("smoke")


def test_too_few_val_texts_refused(repo: Path) -> None:
    (paths.data_prepared() / "metadata_val.csv").write_text(VAL[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="need 5 test sentences"):
        train_config.build("smoke")


def test_malformed_val_line_refused(repo: Path) -> None:
    (paths.data_prepared() / "metadata_val.csv").write_text(
        "\n".join([*VAL[:4], "v9|only two"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"lines \[5\]"):
        train_config.build("smoke")


def test_author_paths_compares_whole_folders(repo: Path) -> None:
    sibling = str(paths.root()) + "2"  # /x/drive2 is not inside /x/drive
    assert train_config.author_paths({"a": str(paths.runs()), "b": sibling + "/x"}) == {
        "b": sibling + "/x"}


def test_decisions_table_replaced_between_markers(repo: Path) -> None:
    train_config.build("smoke")
    train_config.build("smoke")  # idempotent: one table, surrounding text kept
    text = (repo / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    assert text.count(train_config.DECISIONS_BEGIN) == 1 and "old" not in text
    assert "keep me\n" in text and "keep me too" in text
    assert "| `lr_gen` | 0.0002 | 0.0001 |" in text
    assert "| `batch_size` | 48 | 16 |" in text
    report = (repo / train_config.REPORT_REL).read_text(encoding="utf-8")
    assert "| `lr_gen` | 0.0002 | 0.0001 |" in report


def test_decisions_without_markers_refused(repo: Path) -> None:
    (repo / "docs" / "DECISIONS.md").write_text("# Decisions\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker"):
        train_config.build("smoke")


def test_cli_build_config(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["build-config", "--run", "smoke"]) == 0
    out = capsys.readouterr().out
    assert "PASS: config written" in out
    assert f"=== BEGIN {train_config.CONFIG_REL} ===" in out
    assert f"=== BEGIN {train_config.REPORT_REL} ===" in out
    (paths.pretrained() / "male" / "config.json").unlink()
    assert cli.main(["build-config", "--run", "smoke"]) == 1
    assert "FAIL: build-config" in capsys.readouterr().out


def _fake_coqui(monkeypatch: pytest.MonkeyPatch, load_json: object) -> None:
    """Install a fake TTS.tts.configs.vits_config.VitsConfig (coqui is not installed locally)."""
    import sys
    import types

    class VitsConfig:
        def load_json(self, path: str) -> None:
            load_json(self, path)

    mod = types.ModuleType("TTS.tts.configs.vits_config")
    mod.VitsConfig = VitsConfig  # type: ignore[attr-defined]
    for name in ("TTS", "TTS.tts", "TTS.tts.configs"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "TTS.tts.configs.vits_config", mod)


def test_validate_config_tool(repo: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    import types

    import validate_config

    train_config.build("smoke")

    def ok(self: object, path: str, drop: str = "") -> None:
        """Like coqpit: known fields only (restore_path is not one), nested configs as objects."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if key not in ("restore_path", drop):
                setattr(self, key, value)
        self.audio = types.SimpleNamespace(**data["audio"])  # type: ignore[attr-defined]
        self.datasets = [types.SimpleNamespace(**d) for d in data["datasets"]]  # type: ignore

    _fake_coqui(monkeypatch, ok)
    assert validate_config.main() == 0
    out = capsys.readouterr().out
    assert "sample_rate 22050, batch_size 16, lr_gen 0.0001" in out
    assert "PASS: coqui accepts config" in out

    _fake_coqui(monkeypatch, lambda self, path: ok(self, path, drop="save_best_after"))
    assert validate_config.main() == 1
    assert "not these overrides: ['save_best_after']" in capsys.readouterr().out

    def bad(self: object, path: str) -> None:
        raise TypeError("unknown field")

    _fake_coqui(monkeypatch, bad)
    assert validate_config.main() == 1
    assert "FAIL: coqui rejects" in capsys.readouterr().out
