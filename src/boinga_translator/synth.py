"""Voice bank: the baseline and fine-tuned (boinga) voices, loaded once, speaking prepared text."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from boinga_translator import paths, pretrained, settings, synth_basic, text_frontend, train_config

Loader = Callable[[Path, Path], Any]  # (model.pth, config.json) -> coqui-like Synthesizer


def default_voices() -> dict[str, Path]:
    """voice name -> folder holding model.pth + config.json (absent folders are not available)."""
    return {"baseline": train_config.base_dir(), "boinga": paths.best_ckpt()}


def _files(folder: Path) -> tuple[Path, Path] | None:
    model, config = folder / pretrained.MODEL_FILE, folder / pretrained.CONFIG_FILE
    return (model, config) if model.is_file() and config.is_file() else None


class VoiceBank:
    """Named voices; each Synthesizer is loaded on first use and kept for later calls.

    Every text goes through text_frontend.prepare_text (NFC + configs/char_map.json), the same
    front-end the training manifests used. Inference scales come from settings (load_synth).
    """

    def __init__(self, loader: Loader | None = None,
                 voices: dict[str, Path] | None = None) -> None:
        self._loader = loader or synth_basic.load_synth
        self._folders = dict(voices) if voices is not None else default_voices()
        self._synths: dict[str, Any] = {}

    def available(self) -> list[str]:
        """Voices whose folder holds both model.pth and config.json, in bank order."""
        return [name for name, folder in self._folders.items() if _files(folder)]

    def _synth(self, voice: str) -> Any:
        if voice not in self._folders:
            raise ValueError(f"unknown voice {voice!r}: choose one of {list(self._folders)}")
        if voice not in self._synths:
            files = _files(self._folders[voice])
            if files is None:
                raise ValueError(f"voice {voice!r} not available: no model.pth + config.json in "
                                 f"{self._folders[voice]}")
            self._synths[voice] = self._loader(*files)
        return self._synths[voice]

    def speak(self, voice: str, text: str,
              seed: int = settings.SEED) -> tuple[int, np.ndarray]:
        """(sample_rate, float32 mono wav) of `text` in `voice`, with a fixed seed."""
        if voice not in self._folders:  # before the text check: the more useful error
            raise ValueError(f"unknown voice {voice!r}: choose one of {list(self._folders)}")
        prepared = text_frontend.prepare_text(text)
        if not prepared:
            raise ValueError(f"text is empty after preparation: {text!r}")
        synth = self._synth(voice)
        synth_basic.set_seed(seed)
        wav = np.asarray(synth.tts(prepared), dtype=np.float32).reshape(-1)
        return int(synth.output_sample_rate), wav

    def speak_to_file(self, voice: str, text: str, out_wav: Path | str,
                      seed: int = settings.SEED) -> float:
        """Speak `text` into a wav file (coqui's save_wav); return its duration in seconds."""
        rate, wav = self.speak(voice, text, seed)
        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._synth(voice).save_wav(wav, str(out))
        return wav.size / rate
