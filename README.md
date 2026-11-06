# boinga-translator

Single-speaker Chittagonian text-to-speech, built by fine-tuning a pretrained
Bangla VITS model (coqui-tts 0.28.0).

- Training runs on free Google Colab (`pip install -e .[train]`).
- A Gradio demo will ship the fine-tuned voice.
- Local dev needs no GPU: `pip install -e .[dev]` then `pytest -q`.

**Status:** work in progress (proof of concept).
