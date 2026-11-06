"""Tests for report_io.py and tools/save_pasted_reports.py: round trip, traversal, big-file skip."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import save_pasted_reports

from boinga_translator import paths, report_io

FILES = {
    "reports/env.json": '{\n  "python": "3.12.13",\n  "gpu": "Tesla T4"\n}\n',
    "reports/vocab.txt": "অ আ ই\nকথা কইলাম\n",  # Bengali script survives the round trip
    "reports/no_newline.txt": "last line without newline",
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point REPO and ROOT at tmp_path so nothing touches the real repo or Drive."""
    monkeypatch.setattr(paths, "REPO", tmp_path / "repo")
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    (tmp_path / "repo").mkdir()
    return tmp_path / "repo"


def _emit_all(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    for rel, text in FILES.items():
        src = tmp_path / "src" / rel
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(text, encoding="utf-8", newline="\n")
        report_io.emit(src, rel)
    return capsys.readouterr().out


def test_round_trip_emit_parse(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = _emit_all(tmp_path, capsys)
    assert "=== BEGIN reports/env.json ===" in out and "=== END reports/env.json ===" in out
    blocks = report_io.parse_blocks(out)
    assert blocks == {
        "reports/env.json": FILES["reports/env.json"],
        "reports/vocab.txt": FILES["reports/vocab.txt"],
        "reports/no_newline.txt": FILES["reports/no_newline.txt"] + "\n",
    }
    assert not any("=== " in content for content in blocks.values())


def test_parse_tolerates_colab_paste_noise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _emit_all(tmp_path, capsys)
    noisy = (
        "Mounted at /content/drive\n\n"
        + out.replace("=== BEGIN", "\n\n  === BEGIN")  # blank lines + indented markers
        .replace(" ===\n", " ===\n\n")  # blank line after every marker
        .replace("=== END", "\n=== END")  # blank line before END
        + "\nPASS: reports emitted\n"
    ).replace("\n", "\r\n")  # Windows clipboard line endings
    assert report_io.parse_blocks(noisy) == report_io.parse_blocks(out)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("=== BEGIN a.txt ===\nx\n", "no END marker"),
        ("=== BEGIN a.txt ===\nx\n=== END b.txt ===\n", "does not match"),
        ("=== BEGIN a.txt ===\n=== BEGIN b.txt ===\n=== END b.txt ===\n", "inside"),
    ],
)
def test_parse_rejects_broken_pastes(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        report_io.parse_blocks(text)


def test_big_file_is_skipped_with_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    big = tmp_path / "big.json"
    big.write_bytes(b"x" * (report_io.MAX_EMIT_BYTES + 1))
    report_io.emit(big, "reports/big.json")
    out = capsys.readouterr().out
    sha = hashlib.sha256(big.read_bytes()).hexdigest()
    assert out.strip() == f"SKIPPED (too big): reports/big.json, sha256={sha}"
    assert "BEGIN" not in out


def test_non_utf8_file_is_skipped_with_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00bad")
    report_io.emit(binary, "reports/blob.bin")
    sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert capsys.readouterr().out.strip() == f"SKIPPED (not utf-8): reports/blob.bin, sha256={sha}"


@pytest.mark.parametrize(
    "bad",
    [
        "../evil.txt", "reports/../../evil.txt", "/etc/passwd", "C:/x.txt", "a\\b.txt", "",
        ".git/hooks/pre-commit", ".claude/settings.json", "src/boinga_translator/paths.py",
        "reports", "reports/a.txt:ads", "reports/NUL", "reports/con.txt", "reports/x. ",
    ],
)
def test_path_traversal_rejected(repo: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe report path"):
        report_io.save_blocks({bad: "x\n"})
    with pytest.raises(ValueError, match="unsafe report path"):
        report_io.write_report("x\n", bad)
    assert list(repo.iterdir()) == []


def test_save_blocks_writes_nothing_if_any_path_is_bad(repo: Path) -> None:
    with pytest.raises(ValueError):
        report_io.save_blocks({"reports/ok.txt": "ok\n", "../evil.txt": "x\n"})
    assert not (repo / "reports" / "ok.txt").exists()


def test_write_report_writes_repo_and_drive_copies(repo: Path) -> None:
    repo_copy, drive_copy = report_io.write_report("hello\n", "reports/env.json")
    assert repo_copy == (repo / "reports" / "env.json").resolve()
    assert drive_copy == (paths.drive_reports() / "reports" / "env.json").resolve()
    assert repo_copy.read_bytes() == drive_copy.read_bytes() == b"hello\n"


def test_write_report_refuses_when_drive_not_mounted_on_colab(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "content" / "drive" / "MyDrive"  # sandboxed stand-in for /content
    monkeypatch.delenv("BOINGA_ROOT")
    monkeypatch.delenv("BOINGA_FAST")
    monkeypatch.setattr(paths, "COLAB_DRIVE", drive)
    monkeypatch.setattr(paths, "COLAB_ROOT", drive / "boinga-translator")
    monkeypatch.setattr(paths, "is_colab", lambda: True)
    with pytest.raises(RuntimeError, match="Drive is not mounted"):
        report_io.write_report("x\n", "reports/env.json")
    assert list(repo.iterdir()) == [] and not (tmp_path / "content").exists()


def test_save_tool_writes_blocks_from_file(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pasted = tmp_path / "pasted.txt"
    pasted.write_text(_emit_all(tmp_path, capsys), encoding="utf-8")
    assert save_pasted_reports.main([str(pasted)]) == 0
    assert (repo / "reports" / "vocab.txt").read_text(encoding="utf-8") == FILES["reports/vocab.txt"]
    assert "saved reports/env.json" in capsys.readouterr().out


def test_save_tool_rejects_traversal(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pasted = tmp_path / "pasted.txt"
    pasted.write_text("=== BEGIN ../evil.txt ===\nx\n=== END ../evil.txt ===\n", encoding="utf-8")
    assert save_pasted_reports.main([str(pasted)]) == 1
    assert "unsafe report path" in capsys.readouterr().err
    assert not (tmp_path / "evil.txt").exists()
