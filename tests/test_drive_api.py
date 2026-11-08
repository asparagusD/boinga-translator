"""Tests for drive_api.py with a fake Drive v3 service (no network, no Colab)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from boinga_translator import drive_api, paths


class FakeFiles:
    """files().list(q=...).execute() over an in-memory tree: id -> (name, parent, mime, extra)."""

    def __init__(self, items: dict[str, tuple[str, str, str, dict[str, Any]]]) -> None:
        self.items, self.queries = items, []

    def list(self, q: str, fields: str, pageSize: int, pageToken: str | None) -> Any:
        self.queries.append(q)
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        name = re.search(r"name = '((?:\\.|[^'\\])*)'", q)
        want = name.group(1).replace("\\'", "'") if name else None
        folders_only = f"mimeType = '{drive_api.FOLDER}'" in q
        no_folders = f"mimeType != '{drive_api.FOLDER}'" in q
        hits = [{"id": i, "name": n, **extra} for i, (n, p, m, extra) in self.items.items()
                if p == parent and (want is None or n == want)
                and (not folders_only or m == drive_api.FOLDER)
                and (not no_folders or m != drive_api.FOLDER)]
        page = 0 if pageToken is None else int(pageToken)
        res = {"files": hits[page:page + 1]}  # one per page: exercises paging
        if page + 1 < len(hits):
            res["nextPageToken"] = str(page + 1)
        return type("Req", (), {"execute": lambda self: res})()


class FakeService:
    def __init__(self, items: dict[str, tuple[str, str, str, dict[str, Any]]]) -> None:
        self._files = FakeFiles(items)

    def files(self) -> FakeFiles:
        return self._files


F = drive_api.FOLDER
TREE = {
    "a": ("boinga-translator", "root", F, {}),
    "b": ("milestones", "a", F, {}),
    "c": ("run1", "b", F, {}),
    "d": ("step_36500", "c", F, {}),
    "m": ("model.pth", "d", "application/x-zip", {"size": "300", "md5Checksum": "abc"}),
    "k": ("config.json", "d", "application/json", {"size": "9", "md5Checksum": "def"}),
    "x": ("model.pth", "c", "application/x-zip", {"size": "1"}),  # wrong folder
}
STEP = paths.COLAB_DRIVE / "boinga-translator" / "milestones" / "run1" / "step_36500"


def test_drive_parts() -> None:
    assert drive_api.drive_parts(STEP) == ["boinga-translator", "milestones", "run1",
                                           "step_36500"]
    with pytest.raises(ValueError, match="not under"):
        drive_api.drive_parts(Path("/content/boinga-fast/runs"))


def test_listing_walks_the_folders() -> None:
    svc = FakeService(TREE)
    assert drive_api.listing(svc, STEP) == {
        "model.pth": {"size": 300, "md5": "abc"}, "config.json": {"size": 9, "md5": "def"}}
    with pytest.raises(FileNotFoundError, match="not on Drive"):
        drive_api.listing(svc, STEP.with_name("step_1"))  # absent folder
    with pytest.raises(FileNotFoundError):
        drive_api.listing(svc, paths.COLAB_DRIVE / "other" / "x")  # absent parent


def test_duplicates_are_refused() -> None:
    twice = {**TREE, "d2": ("step_36500", "c", F, {})}
    with pytest.raises(RuntimeError, match="2 folders named 'step_36500'"):
        drive_api.listing(FakeService(twice), STEP)
    files = {**TREE, "m2": ("model.pth", "d", "application/x-zip", {"size": "1"})}
    with pytest.raises(RuntimeError, match="duplicate files"):
        drive_api.listing(FakeService(files), STEP)


def test_quotes_in_names() -> None:
    svc = FakeService({"a": ("it's", "root", F, {})})
    assert drive_api.folder_id(svc, ["it's"]) == "a"
    assert "name = 'it\\'s'" in svc.files().queries[0]


def test_lister() -> None:
    assert drive_api.lister(FakeService(TREE))(STEP)["model.pth"]["md5"] == "abc"
