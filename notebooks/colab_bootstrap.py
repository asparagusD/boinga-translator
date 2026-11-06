"""Colab bootstrap cells: mount Drive, clone/pull the private repo, install deps, record env."""

# HOW TO USE
# 1. New Colab notebook -> Runtime > Change runtime type > T4 GPU.
# 2. Key icon ("Secrets") -> add GH_TOKEN = a fine-grained, read-only GitHub token for
#    asparagusD/boinga-translator only (Contents: Read-only), and enable notebook access.
# 3. Paste each "# %%" block below into its own cell, in order, and run them top to bottom.
#    In cell 2 set BRANCH to the branch you are testing (default "main").
# 4. Copy the whole output (incl. the === BEGIN/END reports/env.json === block) back to
#    Claude Code. If pip asks you to restart, do Runtime > Restart session, then run 3 and 4.
# The token is never printed and never stored in a git URL: cell 2 puts it in git's credential
# store on this VM only (~/.git-credentials, mode 600), which disappears with the VM.

# %% Cell 1/4 - mount Google Drive
import os

from google.colab import drive

drive.mount("/content/drive")
ok = os.path.isdir("/content/drive/MyDrive")
print("PASS: Drive mounted" if ok else "FAIL: Drive not mounted at /content/drive/MyDrive")

# %% Cell 2/4 - clone or update the private repo with the GH_TOKEN secret
BRANCH = "main"  # set to the branch under test, e.g. "chunk/C1-paths-env"

import os
import pathlib
import subprocess

OWNER, REPO = "asparagusD", "boinga-translator"
REPO_DIR = "/content/boinga-translator"
SETUP_HELP = """FAIL: Colab secret GH_TOKEN is missing or not shared with this notebook.
  1. github.com > Settings > Developer settings > Fine-grained tokens > Generate new token
     Repository access: Only select repositories -> asparagusD/boinga-translator
     Permissions: Contents = Read-only (nothing else)
  2. In Colab, click the key icon (Secrets) > Add new secret: name GH_TOKEN, value = token
  3. Switch on "Notebook access" for GH_TOKEN, then re-run this cell."""

if not os.path.isdir("/content/drive/MyDrive"):
    print("FAIL: run the bootstrap cells first (cell 1 mounts Drive)")
    raise SystemExit(1)
try:
    from google.colab import userdata

    token = userdata.get("GH_TOKEN")
except Exception:  # noqa: BLE001 - SecretNotFoundError / NotebookAccessError
    token = None
if not token:
    print(SETUP_HELP)
    raise SystemExit(1)


def git(*args: str, cwd: str = REPO_DIR) -> str:
    """Run git, hide the token in any output, stop the cell with FAIL on error."""
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    out = (r.stdout + r.stderr).replace(token, "***").strip()
    if r.returncode != 0:
        print(out)
        print(f"FAIL: git {args[0]} failed")
        raise SystemExit(1)
    return out


creds = pathlib.Path.home() / ".git-credentials"
fd = os.open(creds, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # private from the start
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(f"https://x-access-token:{token}@github.com\n")
git("config", "--global", "credential.helper", "store", cwd="/content")
if os.path.isdir(f"{REPO_DIR}/.git"):
    git("checkout", "-q", "--", "reports/")  # drop re-generated reports; Drive keeps a copy
    git("fetch", "-q", "origin")
else:
    git("clone", "-q", f"https://github.com/{OWNER}/{REPO}.git", REPO_DIR, cwd="/content")
git("switch", BRANCH)
git("pull", "-q", "--ff-only")
print(git("log", "-1", "--oneline"))
print(f"PASS: {REPO_DIR} on {BRANCH}")

# %% Cell 3/4 - install pinned dependencies (coqui-tts==0.28.0 via the [train] extra)
import importlib.metadata as md
import os
import subprocess
import sys

REPO_DIR = "/content/boinga-translator"
if not os.path.isdir(REPO_DIR):
    print("FAIL: run the bootstrap cells first")
    raise SystemExit(1)
pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"],
                     cwd=REPO_DIR, check=False)
versions = {}
for dist in ("torch", "coqui-tts", "numpy"):
    try:
        versions[dist] = md.version(dist)
    except md.PackageNotFoundError:
        versions[dist] = None
    print(f"{dist:<10} {versions[dist]}")
if pip.returncode == 0 and all(versions.values()):
    print("PASS: dependencies installed")
else:
    print(f"FAIL: pip exit {pip.returncode}, versions {versions}")
    raise SystemExit(1)

# %% Cell 4/4 - create project folders and record the environment report
import os
import sys

REPO_DIR = "/content/boinga-translator"
if not os.path.isdir(REPO_DIR):
    print("FAIL: run the bootstrap cells first")
    raise SystemExit(1)
os.chdir(REPO_DIR)
if f"{REPO_DIR}/src" not in sys.path:  # editable install is not visible until a restart
    sys.path.insert(0, f"{REPO_DIR}/src")
for name in [m for m in sys.modules if m.split(".")[0] == "boinga_translator"]:
    del sys.modules[name]  # re-import so code pulled by cell 2 is used, not a cached copy
from boinga_translator import env_report, paths

paths.ensure_dirs()
env_report.run()  # prints the reports/env.json block and "PASS: env recorded"
