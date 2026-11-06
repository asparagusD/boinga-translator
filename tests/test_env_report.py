"""Tests for env_report.py and `boinga env`: missing torch, fake GPU, JSON keys, CLI wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from boinga_translator import cli, env_report, paths

KEYS = {
    "python", "platform", "is_colab", "torch", "torch_cuda", "torch_error",
    "gpu_name", "gpu_mem_gb", "coqui_tts", "numpy", "git_commit",
    "disk_free_gb_root", "disk_free_gb_fast", "utc_time",
}


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """REPO, ROOT and FAST all under tmp_path; local mode; no real git repo."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "REPO", repo)
    monkeypatch.setattr(paths, "is_colab", lambda: False)
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # git can't find an outer repo
    monkeypatch.delenv("GIT_DIR", raising=False)
    return repo


def test_torch_absent_does_not_crash(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    env = env_report.run()
    out = capsys.readouterr().out
    assert env["torch"] is None and env["torch_cuda"] is None
    assert env["gpu_name"] == "none" and env["gpu_mem_gb"] is None
    assert "WARN: torch not installed" in out
    assert out.rstrip().endswith("PASS: env recorded")


def test_broken_torch_is_reported_not_raised(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import builtins

    real_import = builtins.__import__

    def broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise OSError("[WinError 126] fbgemm.dll not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    env = env_report.run()
    out = capsys.readouterr().out
    assert env["torch"] is None and env["torch_error"].startswith("OSError: [WinError 126]")
    assert "WARN: torch import failed: OSError" in out and "WARN: torch not installed" not in out
    assert out.rstrip().endswith("PASS: env recorded")


def test_free_gb_returns_none_on_os_error(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stale(path: Path) -> None:
        raise OSError("Transport endpoint is not connected")

    monkeypatch.setattr(env_report.shutil, "disk_usage", stale)
    assert env_report.collect()["disk_free_gb_root"] is None


def test_coqui_absent_warns(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(env_report, "_version", lambda dist: None if dist == "coqui-tts" else "1")
    assert env_report.run()["coqui_tts"] is None
    assert "WARN: coqui-tts not installed" in capsys.readouterr().out


def test_fake_gpu_is_recorded(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    props = SimpleNamespace(name="Tesla T4", total_memory=15_828_320_256)
    fake_torch = SimpleNamespace(
        __version__="2.5.1",
        version=SimpleNamespace(cuda="12.4"),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_properties=lambda i: props),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    env = env_report.run()
    assert (env["torch"], env["torch_cuda"]) == ("2.5.1", "12.4")
    assert (env["gpu_name"], env["gpu_mem_gb"]) == ("Tesla T4", 14.7)


def test_json_keys_present_and_emitted(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = env_report.run()
    saved = json.loads((sandbox / "reports" / "env.json").read_text(encoding="utf-8"))
    assert set(saved) == KEYS == set(env)
    assert saved["git_commit"] == "unknown"  # tmp_path is not a git repo
    assert isinstance(saved["disk_free_gb_root"], float)
    assert saved["utc_time"].endswith("+00:00")
    out = capsys.readouterr().out
    assert "=== BEGIN reports/env.json ===" in out and "=== END reports/env.json ===" in out
    assert (paths.drive_reports() / "reports" / "env.json").is_file()


def test_cli_env_subcommand_calls_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(env_report, "run", lambda *a: calls.append(a) or {})
    assert cli.main(["env"]) == 0
    assert calls == [()]
