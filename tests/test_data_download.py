"""Tests for data_download.py and `boinga fetch-data`: resume, sha256 check, one-time unzip, copy."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from boinga_translator import cli, data_download, paths, settings

FILES = {"Dataset/audio/utt_001.wav": b"RIFF-fake-1", "Dataset/meta.csv": "id,text\n1,হ\n"}


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in FILES.items():
            zf.writestr(name, data)
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class FakeServer:
    """urlopen stand-in serving `data`; honours Range unless `honour_range` is False."""

    def __init__(self, data: bytes, honour_range: bool = True) -> None:
        self.data, self.honour_range = data, honour_range
        self.ranges: list[str | None] = []
        self.agents: list[str | None] = []

    def __call__(self, req: urllib.request.Request) -> _Resp:
        self.agents.append(req.get_header("User-agent"))  # urllib stores it capitalised
        rng = req.get_header("Range")
        self.ranges.append(rng)
        if rng and self.honour_range:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            return _Resp(self.data[start:], 206)
        return _Resp(self.data, 200)


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    monkeypatch.setenv("BOINGA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("BOINGA_FAST", str(tmp_path / "fast"))
    data = _zip_bytes()
    monkeypatch.setattr(settings, "DATASET_ZIP_BYTES", len(data))
    monkeypatch.setattr(settings, "DATASET_SHA256", hashlib.sha256(data).hexdigest())
    fake = FakeServer(data)
    monkeypatch.setattr(data_download, "_open", fake)
    return fake


def _steps(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(("SKIP ", "DONE "))]


def test_first_call_downloads_verifies_unzips_and_copies(
        server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    got = data_download.fetch()
    out = capsys.readouterr().out
    assert got == paths.data_fast()
    assert _steps(out) == ["DONE download", "DONE verify", "DONE unzip", "DONE copy"]
    assert f"PASS: dataset ready at {paths.data_fast()}, 2 files" in out
    assert (paths.data_fast() / "Dataset/meta.csv").read_text(encoding="utf-8") == FILES[
        "Dataset/meta.csv"]
    assert (paths.data_unzipped() / "Dataset/audio/utt_001.wav").is_file()
    assert (paths.data_raw() / "Dataset.zip.sha256_ok").is_file()
    assert server.ranges == [None]


def test_requests_send_project_user_agent(server: FakeServer) -> None:
    """Mendeley's Cloudflare answers 403 to urllib's default Python-urllib agent (C3.V1)."""
    part = paths.data_raw() / "Dataset.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(server.data[:10])
    data_download.fetch()
    assert server.agents == [data_download.USER_AGENT]
    assert data_download.USER_AGENT.startswith("boinga-translator/")


def test_second_call_skips_all(server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    data_download.fetch()
    capsys.readouterr()
    data_download.fetch()
    out = capsys.readouterr().out
    assert _steps(out) == ["SKIP download (done)", "SKIP verify (done)", "SKIP unzip (done)",
                           "SKIP copy (done)"]
    assert len(server.ranges) == 1
    assert "PASS: dataset ready at" in out


def test_deleted_fast_copy_is_recopied_only(
        server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    data_download.fetch()
    data_download.shutil.rmtree(paths.data_fast())
    capsys.readouterr()
    data_download.fetch()
    out = capsys.readouterr().out
    assert _steps(out) == ["SKIP download (done)", "SKIP verify (done)", "SKIP unzip (done)",
                           "DONE copy"]
    assert (paths.data_fast() / "Dataset/audio/utt_001.wav").is_file()
    assert len(server.ranges) == 1


def test_sha_mismatch_raises_with_both_values(
        server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "DATASET_SHA256", "0" * 64)
    real = hashlib.sha256(server.data).hexdigest()
    with pytest.raises(RuntimeError, match=f"expected {'0' * 64}.*got {real}"):
        data_download.fetch()
    assert not (paths.data_raw() / "Dataset.zip").exists()
    assert not (paths.data_raw() / "Dataset.zip.sha256_ok").exists()


def test_size_mismatch_raises_with_both_values(
        server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "DATASET_ZIP_BYTES", len(server.data) - 7)
    with pytest.raises(RuntimeError, match=f"expected {len(server.data) - 7}.*got "
                                           f"{len(server.data)}"):
        data_download.fetch()
    assert not (paths.data_raw() / "Dataset.zip").exists()


def test_partial_file_resumes_with_range(server: FakeServer) -> None:
    part = paths.data_raw() / "Dataset.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(server.data[:10])
    data_download.fetch()
    assert server.ranges == ["bytes=10-"]
    assert (paths.data_raw() / "Dataset.zip").read_bytes() == server.data
    assert not part.exists()


def test_server_ignoring_range_restarts_cleanly(
        server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    server.honour_range = False
    part = paths.data_raw() / "Dataset.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(server.data[:10])
    data_download.fetch()
    assert server.ranges == ["bytes=10-"]
    assert (paths.data_raw() / "Dataset.zip").read_bytes() == server.data
    assert "restarting" in capsys.readouterr().out


def test_oversized_partial_file_restarts(server: FakeServer) -> None:
    part = paths.data_raw() / "Dataset.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(server.data + b"junk")
    data_download.fetch()
    assert server.ranges == [None]
    assert (paths.data_raw() / "Dataset.zip").read_bytes() == server.data


def test_partial_unzip_is_redone(server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    data_download.fetch()
    (paths.data_unzipped() / ".unzipped_ok").unlink()
    (paths.data_unzipped() / "stray.tmp").write_bytes(b"x")
    capsys.readouterr()
    data_download.fetch()
    out = capsys.readouterr().out
    assert "DONE unzip" in out and "DONE copy" in out
    assert not (paths.data_unzipped() / "stray.tmp").exists()
    assert not (paths.data_fast() / "stray.tmp").exists()


def test_deleted_zip_redoes_every_step(
        server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    data_download.fetch()
    (paths.data_raw() / "Dataset.zip").unlink()
    capsys.readouterr()
    data_download.fetch()
    assert _steps(capsys.readouterr().out) == ["DONE download", "DONE verify", "DONE unzip",
                                               "DONE copy"]
    assert len(server.ranges) == 2


def test_short_stream_keeps_part_and_rerun_resumes(server: FakeServer) -> None:
    full = server.data
    server.data = full[:15]  # connection ends early without an error
    with pytest.raises(RuntimeError, match=f"stopped at 15 of {len(full)} B"):
        data_download.fetch()
    part = paths.data_raw() / "Dataset.zip.part"
    assert part.read_bytes() == full[:15]
    assert not (paths.data_raw() / "Dataset.zip").exists()
    server.data = full
    data_download.fetch()
    assert server.ranges == [None, "bytes=15-"]
    assert (paths.data_raw() / "Dataset.zip").read_bytes() == full


def test_range_not_satisfiable_deletes_part(
        server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    part = paths.data_raw() / "Dataset.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x" * 10)

    def refuse(req: urllib.request.Request) -> _Resp:
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)

    monkeypatch.setattr(data_download, "_open", refuse)
    with pytest.raises(RuntimeError, match="416"):
        data_download.fetch()
    assert not part.exists()


def test_cli_fetch_data(server: FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["fetch-data"]) == 0
    assert "PASS: dataset ready at" in capsys.readouterr().out


def test_cli_fetch_data_fails_loudly(
        server: FakeServer, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(settings, "DATASET_SHA256", "f" * 64)
    assert cli.main(["fetch-data"]) == 1
    assert "FAIL: fetch-data:" in capsys.readouterr().out
