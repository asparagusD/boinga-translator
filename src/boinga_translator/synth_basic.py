"""Minimal seeded synthesis for any VITS checkpoint: load a coqui Synthesizer, say text to a wav."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from boinga_translator import settings

# model attribute -> pinned value (settings.py); set only when the model has the attribute
INFERENCE_SCALES: dict[str, float] = {
    "inference_noise_scale": settings.INFER_NOISE_SCALE,
    "inference_noise_scale_dp": settings.INFER_NOISE_SCALE_W,
    "length_scale": settings.INFER_LENGTH_SCALE,
}


def set_seed(seed: int = settings.SEED) -> None:
    """Seed random, numpy and (if installed) torch incl. all CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_synth(model_path: Path | str, config_path: Path | str,
               use_cuda: bool | None = None) -> Any:
    """coqui Synthesizer for a checkpoint, with the pinned inference scales applied."""
    from TTS.utils.synthesizer import Synthesizer

    if use_cuda is None:
        import torch

        use_cuda = bool(torch.cuda.is_available())
    synth = Synthesizer(tts_checkpoint=str(model_path), tts_config_path=str(config_path),
                        use_cuda=use_cuda)
    model = synth.tts_model
    applied = [name for name in INFERENCE_SCALES if hasattr(model, name)]
    for name in applied:
        setattr(model, name, INFERENCE_SCALES[name])
    missing = [name for name in INFERENCE_SCALES if name not in applied]
    print(f"synth: use_cuda={use_cuda}; set "
          + ", ".join(f"{n}={INFERENCE_SCALES[n]}" for n in applied)
          + (f"; not on model: {', '.join(missing)}" if missing else ""))
    return synth


def say(synth: Any, text: str, out_wav: Path | str, seed: int = settings.SEED) -> float:
    """Synthesise text with a fixed seed, save it as a wav, return its duration in seconds.

    The duration is that of the saved wav (list or 1-D array), so it includes any silence coqui's
    Synthesizer.tts() appends after each sentence. On CUDA the same seed gives the same noise but
    not necessarily bit-identical audio (cuDNN kernels), so compare by listening, not by bytes.
    """
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    wav = synth.tts(text)
    synth.save_wav(wav, str(out))
    return len(wav) / synth.output_sample_rate
