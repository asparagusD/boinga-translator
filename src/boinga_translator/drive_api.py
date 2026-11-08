"""Drive API listing of My Drive folders: checks what Drive holds, not what the mount shows."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from boinga_translator import paths

FOLDER = "application/vnd.google-apps.folder"


def service() -> Any:
    """Drive v3 client for the Colab user (sign-in popup); Colab only."""
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    return build("drive", "v3", cache_discovery=False)


def drive_parts(path: Path) -> list[str]:
    """Folder names from My Drive down to path (a path under the Colab Drive mount)."""
    try:
        return list(Path(path).relative_to(paths.COLAB_DRIVE).parts)
    except ValueError:
        raise ValueError(f"{path} is not under {paths.COLAB_DRIVE}") from None


def _quote(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def _list(svc: Any, query: str, fields: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    token = None
    while True:
        res = svc.files().list(q=query, fields=f"nextPageToken, files({fields})",
                               pageSize=100, pageToken=token).execute()
        files += res.get("files", [])
        if not (token := res.get("nextPageToken")):
            return files


def folder_id(svc: Any, parts: list[str]) -> str | None:
    """Id of the folder My Drive/<parts...>, or None if any level is missing."""
    parent = "root"
    for part in parts:
        found = _list(svc, f"'{parent}' in parents and name = '{_quote(part)}' and "
                           f"mimeType = '{FOLDER}' and trashed = false", "id")
        if not found:
            return None
        if len(found) > 1:
            raise RuntimeError(f"{len(found)} folders named {part!r} on Drive "
                               f"({[f['id'] for f in found]}): merge or rename them")
        parent = found[0]["id"]
    return parent


def listing(svc: Any, folder: Path) -> dict[str, dict[str, Any]]:
    """name -> {size, md5} of the files Drive holds in a mounted folder.

    FileNotFoundError if the folder (or a parent) is not on Drive; RuntimeError if a name is
    taken twice (Drive allows duplicates; the check would compare the wrong copy).
    """
    parts = drive_parts(folder)
    fid = folder_id(svc, parts)
    if fid is None:
        raise FileNotFoundError(f"{folder} is not on Drive (My Drive/{'/'.join(parts)})")
    files = _list(svc, f"'{fid}' in parents and trashed = false and mimeType != '{FOLDER}'",
                  "name, size, md5Checksum")
    names = [f["name"] for f in files]
    if dup := sorted({n for n in names if names.count(n) > 1}):
        raise RuntimeError(f"duplicate files {dup} in {folder} on Drive: delete the extra copies")
    return {f["name"]: {"size": int(f["size"]) if "size" in f else None,
                        "md5": f.get("md5Checksum")} for f in files}


def lister(svc: Any | None = None) -> Callable[[Path], dict[str, dict[str, Any]]]:
    """A folder -> listing function for milestones.check (signs in on first use)."""
    if svc is None:
        try:
            svc = service()
            user = svc.about().get(fields="user(emailAddress)").execute()["user"]
        except Exception as exc:  # popup dismissed, missing libs: one type for the CLI
            raise RuntimeError(f"Drive API sign-in failed: {exc!r}") from exc
        print(f"Drive API as {user.get('emailAddress')}: must own the mounted Drive")

    def list_folder(folder: Path) -> dict[str, dict[str, Any]]:
        try:
            return listing(svc, folder)
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:  # googleapiclient HttpError etc.: one type for the CLI
            raise RuntimeError(f"Drive API listing of {folder} failed: {exc!r}") from exc

    return list_folder
