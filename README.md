# boinga-translator

Single-speaker Chittagonian text-to-speech, built by fine-tuning a pretrained
Bangla VITS model (coqui-tts 0.28.0).

- Training runs on free Google Colab (`pip install -e .[train]`).
- A Gradio demo will ship the fine-tuned voice.
- Local dev needs no GPU: `pip install -e .[dev]` then `pytest -q`.

**Status:** work in progress (proof of concept).

## Developer setup

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"   # .venv/bin/python on Linux
git config core.hooksPath .githooks   # once per clone: commit-msg + pre-commit guards
```

The hooks enforce the commit format, block audio/checkpoint/archive files and
anything over 2 MB, and require a fresh `docs/REPO_GRAPH.md`, clean `ruff` and
passing `pytest`.
