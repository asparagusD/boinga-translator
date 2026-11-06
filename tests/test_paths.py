"""Tests for paths.py: env overrides, Colab/local defaults, run_dir and ensure_dirs."""

from __future__ import annotations

from pathlib import Path

import pytest

from boinga_translator import paths


@pytest.fixture
def env_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root, fast = tmp_path / "root", tmp_path / "fast"
    monkeypatch.setenv("BOINGA_ROOT", str(root))
    monkeypatch.setenv("BOINGA_FAST", str(fast))
    return root, fast


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOINGA_ROOT", raising=False)
    monkeypatch.delenv("BOINGA_FAST", raising=False)


def test_is_colab_false_locally() -> None:
    assert paths.is_colab() is False


def test_repo_is_the_repo_root() -> None:
    assert (paths.REPO / "pyproject.toml").is_file()
    assert paths.reports_dir() == paths.REPO / "reports"
    assert paths.configs_dir() == paths.REPO / "configs"


def test_local_defaults(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    assert paths.root() == paths.REPO / "local"
    assert paths.fast() == paths.REPO / "local" / "fast"
    assert paths.ROOT == paths.root() and paths.FAST == paths.fast()


def test_colab_defaults(no_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    assert paths.root().as_posix() == "/content/drive/MyDrive/boinga-translator"
    assert paths.fast().as_posix() == "/content/boinga-fast"


def test_env_overrides(env_roots: tuple[Path, Path]) -> None:
    root, fast = env_roots
    assert paths.ROOT == root and paths.FAST == fast
    expected = {
        paths.data_raw: root / "data" / "raw",
        paths.data_unzipped: root / "data" / "unzipped",
        paths.data_fast: fast / "data",
        paths.data_prepared: fast / "prepared",
        paths.prepared_backup: root / "data" / "prepared",
        paths.pretrained: root / "pretrained",
        paths.runs: root / "runs",
        paths.best_ckpt: root / "best",
        paths.eval_dir: root / "eval",
        paths.drive_reports: root / "reports",
    }
    for fn, path in expected.items():
        assert fn() == path, fn.__name__


def test_fast_follows_root_override_locally(
    tmp_path: Path, no_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path))
    assert paths.fast() == tmp_path / "fast"


def test_run_dir_composes(env_roots: tuple[Path, Path]) -> None:
    root, _ = env_roots
    assert paths.run_dir("smoke-50") == root / "runs" / "smoke-50"


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b"])
def test_run_dir_rejects_non_simple_names(env_roots: tuple[Path, Path], bad: str) -> None:
    with pytest.raises(ValueError):
        paths.run_dir(bad)


def test_ensure_dirs_creates_everything(env_roots: tuple[Path, Path]) -> None:
    created = paths.ensure_dirs()
    assert set(created) == set(paths.all_dirs())
    assert all(d.is_dir() for d in created)


def test_ensure_dirs_refuses_unmounted_drive_on_colab(
    tmp_path: Path, no_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "content" / "drive" / "MyDrive"  # never touch the real /content
    monkeypatch.setattr(paths, "COLAB_DRIVE", drive)
    monkeypatch.setattr(paths, "COLAB_ROOT", drive / "boinga-translator")
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    with pytest.raises(RuntimeError, match="Drive is not mounted"):
        paths.ensure_dirs()
    assert not (tmp_path / "content").exists()
