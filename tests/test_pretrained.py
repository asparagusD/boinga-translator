"""Tests for pretrained.py and `boinga fetch-pretrained`: cache hit, download, fallback, bad input."""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import pytest

from boinga_translator import cli, paths, pretrained, settings


@pytest.fixture(autouse=True)
def _tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))


def _fake_download(files: dict[str, bytes], calls: list[str]) -> pretrained.Downloader:
    """Downloader that writes `files` (relative paths) into a fresh folder."""
    def download(voice: str, workdir: Path) -> Path:
        calls.append(voice)
        out = workdir / "dl"
        for rel, data in files.items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(data)
        return out
    return download


def _failing(voice: str, workdir: Path) -> Path:
    raise RuntimeError("model manager down")


def test_cache_hit_skips_download() -> None:
    target = paths.pretrained() / "female"
    target.mkdir(parents=True)
    (target / "model.pth").write_bytes(b"w")
    (target / "config.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []
    got = pretrained.fetch("female", downloaders=[_fake_download({}, calls)])
    assert calls == []
    assert got == {"model": target / "model.pth", "config": target / "config.json"}


def test_empty_cached_file_triggers_download() -> None:
    target = paths.pretrained() / "male"
    target.mkdir(parents=True)
    (target / "model.pth").write_bytes(b"")
    (target / "config.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []
    pretrained.fetch("male", downloaders=[_fake_download(
        {"model_file.pth": b"weights", "config.json": b"{}"}, calls)])
    assert calls == ["male"]
    assert (target / "model.pth").read_bytes() == b"weights"


def test_download_copies_single_pth_as_model_pth(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []
    got = pretrained.fetch("female", downloaders=[_fake_download(
        {"sub/model_file.pth": b"x" * 2048, "sub/config.json": b'{"a": 1}'}, calls)])
    assert got["model"].read_bytes() == b"x" * 2048
    assert got["config"].read_text(encoding="utf-8") == '{"a": 1}'
    out = capsys.readouterr().out
    assert "MB" in out and "PASS: female cached" in out
    assert not list(paths.pretrained().glob("female/*.part"))


def test_falls_back_to_next_downloader(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []
    pretrained.fetch("male", downloaders=[_failing, _fake_download(
        {"model.pth": b"w", "config.json": b"{}"}, calls)])
    assert calls == ["male"]
    assert "WARN" in capsys.readouterr().out


def test_all_downloaders_fail_raises() -> None:
    with pytest.raises(RuntimeError, match="could not download female"):
        pretrained.fetch("female", downloaders=[_failing, _failing])


def test_ambiguous_pth_is_refused() -> None:
    files = {"a.pth": b"1", "b.pth": b"2", "config.json": b"{}"}
    with pytest.raises(RuntimeError, match=r"^could not download female: .*one \*\.pth"):
        pretrained.fetch("female", downloaders=[_fake_download(files, [])])


def test_release_zip_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zpath = tmp_path / "release.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("tts_models--bn--custom--vits_male/model_file.pth", "weights")
        zf.writestr("tts_models--bn--custom--vits_male/config.json", "{}")
    monkeypatch.setitem(settings.PRETRAINED_RELEASE_URLS, "male", zpath.as_uri())
    got = pretrained.fetch("male", downloaders=[pretrained._download_release])
    assert got["model"].read_text() == "weights"
    assert got["config"].read_text() == "{}"


def test_model_manager_call_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeManager:
        def __init__(self, **kwargs: object) -> None:
            seen["kwargs"] = kwargs

        def download_model(self, name: str) -> tuple[str, str, dict[str, str]]:
            seen["name"] = name
            out = Path(str(seen["kwargs"]["output_prefix"])) / "m"
            out.mkdir()
            (out / "model_file.pth").write_bytes(b"w")
            (out / "config.json").write_text("{}")
            return str(out / "model_file.pth"), str(out / "config.json"), {}

    manage = types.ModuleType("TTS.utils.manage")
    manage.ModelManager = FakeManager  # type: ignore[attr-defined]
    for name, mod in (("TTS", types.ModuleType("TTS")),
                      ("TTS.utils", types.ModuleType("TTS.utils")), ("TTS.utils.manage", manage)):
        monkeypatch.setitem(sys.modules, name, mod)
    got = pretrained.fetch("female", downloaders=[pretrained._download_with_manager])
    assert seen["name"] == settings.PRETRAINED_IDS["female"]
    assert seen["kwargs"]["progress_bar"] is False  # type: ignore[index]
    assert got["model"].read_bytes() == b"w"


def test_drive_guard_refuses_unmounted_drive(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drive = tmp_path / "drive" / "MyDrive"  # never created: Drive not mounted
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    monkeypatch.setattr(paths, "COLAB_DRIVE", drive)
    monkeypatch.setenv("BOINGA_ROOT", str(drive / "boinga-translator"))
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="Drive is not mounted"):
        pretrained.fetch("female", downloaders=[_fake_download({}, calls)])
    assert calls == [] and not (tmp_path / "drive").exists()


def test_drive_guard_ignores_roots_outside_drive(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    monkeypatch.setattr(paths, "COLAB_DRIVE", tmp_path / "drive" / "MyDrive")
    got = pretrained.fetch("female", downloaders=[_fake_download(
        {"model.pth": b"w", "config.json": b"{}"}, [])])
    assert got["model"].is_file()


def test_unknown_voice_raises() -> None:
    with pytest.raises(ValueError, match="unknown voice"):
        pretrained.fetch("robot", downloaders=[_fake_download({}, [])])


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe path"):
        pretrained.safe_extract(zpath, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_unzips(tmp_path: Path) -> None:
    zpath = tmp_path / "ok.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("voice/config.json", "{}")
    pretrained.safe_extract(zpath, tmp_path / "out")
    assert (tmp_path / "out" / "voice" / "config.json").read_text() == "{}"


def test_cli_fetch_pretrained_both(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(pretrained, "fetch", lambda voice, **_: seen.append(voice) or {})
    assert cli.main(["fetch-pretrained", "--voice", "both"]) == 0
    assert seen == ["female", "male"]


def test_cli_fetch_pretrained_failure_exits_1(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(voice: str, **_: object) -> dict[str, Path]:
        raise RuntimeError("no network")
    monkeypatch.setattr(pretrained, "fetch", boom)
    assert cli.main(["fetch-pretrained", "--voice", "male"]) == 1
    assert "FAIL: male: no network" in capsys.readouterr().out
