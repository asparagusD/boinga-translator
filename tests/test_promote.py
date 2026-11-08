"""Tests for train_runner.promote and `boinga promote` (fake checkpoints on a fake Drive)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from boinga_translator import ckpt_sampler, cli, paths, pretrained, settings, train_runner

REAL_REPO = paths.REPO
FOLDER = "run1-November-08-2026_05+25AM-853ac3a"
LOG = "".join(
    f"\n   --> TIME: x -- GLOBAL_STEP: {s}\n\n   --> EVAL PERFORMANCE\n"
    f"     | > avg_loss_0: 2.0\n     | > avg_loss_1: 40.0\n     | > avg_loss_mel: {mel}\n"
    for s, mel in ((10000, 21.1), (20025, 20.9), (36475, 20.6)))


def _ckpt(step: int) -> bytes:
    return f"checkpoint {step}".encode() * 100


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """run1 with checkpoints 10000/20000/36500, config and log; repo configs copied."""
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "drive"))
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    for rel in ("configs/speaker_selection.json", "configs/char_map.json"):
        (paths.REPO / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL_REPO / rel, paths.REPO / rel)
    folder = paths.run_dir("run1") / FOLDER
    folder.mkdir(parents=True)
    for step in (10000, 20000, 36500):
        (folder / f"checkpoint_{step}.pth").write_bytes(_ckpt(step))
    (folder / pretrained.CONFIG_FILE).write_text('{"run_name": "run1"}', encoding="utf-8")
    (folder / ckpt_sampler.LOG_FILE).write_text(LOG, encoding="utf-8")
    return folder


def _prov() -> dict:
    return json.loads((paths.best_ckpt() / train_runner.PROVENANCE).read_text(encoding="utf-8"))


def test_promote_copies_model_config_and_provenance(run: Path) -> None:
    prov = train_runner.promote("run1", 36500)
    best = paths.best_ckpt()
    assert (best / pretrained.MODEL_FILE).read_bytes() == _ckpt(36500)
    assert (best / pretrained.CONFIG_FILE).read_text(encoding="utf-8") == '{"run_name": "run1"}'
    assert not list(best.glob("*.part"))
    assert _prov() == prov
    assert prov["run"] == "run1" and prov["step"] == 36500 and prov["run_folder"] == FOLDER
    assert prov["source"] == "checkpoint_36500.pth"
    assert prov["md5"] == hashlib.md5(_ckpt(36500)).hexdigest()
    assert prov["bytes"] == len(_ckpt(36500))


def test_provenance_fields(run: Path) -> None:
    prov = train_runner.promote("run1", 36500)
    assert prov["eval"] == {"step": 36475, "loss_mel": 20.6, "loss_total": 42.0}
    sel = json.loads((paths.REPO / "configs/speaker_selection.json").read_text(encoding="utf-8"))
    voice = sel["base_voice"]
    assert prov["base_model"] == {"voice": voice, "id": settings.PRETRAINED_IDS[voice],
                                  "license": settings.PRETRAINED_LICENSE,
                                  "coqui_tts": settings.COQUI_TTS_VERSION}
    assert prov["dataset"] == {"name": settings.DATASET_NAME, "doi": settings.DATASET_DOI,
                               "license": settings.DATASET_LICENSE}
    assert prov["speaker_selection"] == {"kind": sel["kind"], "id": sel["id"],
                                         "csv_sha256": sel["csv_sha256"]}
    assert prov["seed"] == settings.SEED
    assert prov["inference"]["length_scale"] == settings.INFER_LENGTH_SCALE
    assert len(prov["char_map"]) == 10
    assert prov["git_commit"] and prov["promoted_utc"].endswith("+00:00")


def test_provenance_emitted_to_repo(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train_runner.promote("run1", 36500)
    out = capsys.readouterr().out
    assert "=== BEGIN reports/best_provenance.json ===" in out
    assert "replaces: nothing" in out
    assert out.rstrip().endswith("PASS: promoted run1 step 36500")
    repo_copy = json.loads((paths.REPO / train_runner.PROVENANCE_REL).read_text(encoding="utf-8"))
    assert repo_copy == _prov()


def test_promote_again_copies_nothing(run: Path, capsys: pytest.CaptureFixture[str],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    first = train_runner.promote("run1", 36500)
    capsys.readouterr()

    def no_copy(*_: object) -> str:
        raise AssertionError("copied again")

    monkeypatch.setattr(train_runner, "_copy_md5", no_copy)
    assert train_runner.promote("run1", 36500) == first
    out = capsys.readouterr().out
    assert "already holds run run1 step 36500" in out
    assert "PASS: promoted run1 step 36500" in out
    assert "=== BEGIN reports/best_provenance.json ===" in out


def test_promote_other_step_says_what_it_replaces(run: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    train_runner.promote("run1", 20000)
    capsys.readouterr()
    train_runner.promote("run1", 36500)
    out = capsys.readouterr().out
    assert f"replaces: run run1 step 20000 ({FOLDER})" in out
    assert (paths.best_ckpt() / pretrained.MODEL_FILE).read_bytes() == _ckpt(36500)
    assert _prov()["step"] == 36500


def test_promote_recopies_when_size_differs(run: Path) -> None:
    train_runner.promote("run1", 36500)
    (paths.best_ckpt() / pretrained.MODEL_FILE).write_bytes(b"truncated")
    train_runner.promote("run1", 36500)
    assert (paths.best_ckpt() / pretrained.MODEL_FILE).read_bytes() == _ckpt(36500)


def test_eval_is_null_without_a_log(run: Path) -> None:
    (run / ckpt_sampler.LOG_FILE).unlink()
    assert train_runner.promote("run1", 36500)["eval"] is None


def test_promote_over_unknown_model(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths.best_ckpt().mkdir(parents=True)
    (paths.best_ckpt() / pretrained.MODEL_FILE).write_bytes(b"old")
    train_runner.promote("run1", 36500)
    out = capsys.readouterr().out
    assert "replaces: a model.pth without provenance" in out
    assert "goes to Drive's trash" in out


def test_missing_step_refused(run: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no checkpoint_36000.pth .*10000, 20000, 36500"):
        train_runner.promote("run1", 36000)
    assert not paths.best_ckpt().exists() or not any(paths.best_ckpt().iterdir())


def test_missing_run_refused(run: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint under"):
        train_runner.promote("run2", 36500)


def test_no_drive_space_refused(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_runner, "free_gb", lambda _: 0.0)
    with pytest.raises(RuntimeError, match="GB free"):
        train_runner.promote("run1", 36500)
    assert not (paths.best_ckpt() / pretrained.MODEL_FILE).exists()


def test_cli_promote(run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["promote", "--run", "run1", "--step", "36500"]) == 0
    assert "PASS: promoted run1 step 36500" in capsys.readouterr().out
    assert cli.main(["promote", "--run", "run1", "--step", "123"]) == 1
    assert "FAIL: promote FileNotFoundError" in capsys.readouterr().out
