# boinga-translator

Single-speaker Chittagonian text-to-speech, built by fine-tuning a pretrained
Bangla VITS model (coqui-tts 0.28.0).

- Training runs on free Google Colab (`pip install -e .[train]`).
- A Gradio demo will ship the fine-tuned voice.
- Local dev needs no GPU: `pip install -e .[dev]` then `pytest -q`.

**Status:** work in progress (proof of concept).

## Run on Colab

Open the bootstrap notebook (mount Drive, clone, install pinned deps, record the env):
[notebooks/runs/C1_bootstrap_run1.ipynb](https://colab.research.google.com/github/asparagusD/boinga-translator/blob/main/notebooks/runs/C1_bootstrap_run1.ipynb).
The repo is private, so Colab must be allowed to read it (File > Open notebook > GitHub,
tick "Include private repos"). Add a `GH_TOKEN` Colab secret first; the clone cell
(BOOT.2) explains how if it is missing. Every notebook in `notebooks/runs/` is generated
from a spec in `notebooks/specs/` by `python tools/build_notebook.py --all`; the bootstrap
cells come from `notebooks/colab_bootstrap.py`.

## Developer setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"   # .venv/bin/python on Linux
git config core.hooksPath .githooks   # once per clone: commit-msg + pre-commit guards
```

The hooks enforce the commit format, block audio/checkpoint/archive files and
anything over 2 MB, and require a fresh `docs/REPO_GRAPH.md`, clean `ruff` and
passing `pytest`.
