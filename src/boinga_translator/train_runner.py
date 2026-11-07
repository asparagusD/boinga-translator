"""Training runner: coqui VITS loader check now; training, resume and throughput follow (C7.3/4)."""

from __future__ import annotations

import contextlib
import inspect
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from boinga_translator import paths, train_config

# Keyword arguments get_data_loader may take: 0.28 has config, *, is_eval, samples, verbose;
# assets/num_gpus are older coqui names. Only those in the printed signature are passed.
_LOADER_KWARGS = ("is_eval", "samples", "verbose", "assets", "num_gpus")


def _np(value: Any) -> np.ndarray:
    """A torch tensor or array-like as a numpy array (on the CPU)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def judge_batch(batch: Mapping[str, Any], vocab_size: int,
                sample_rate: int) -> tuple[list[str], list[str]]:
    """(report lines, problems) for one batch: token ids in [0, vocab), floats finite."""
    lines: list[str] = []
    problems: list[str] = []
    if "tokens" not in batch:
        problems.append("batch has no tokens")
    for key, value in batch.items():
        if not hasattr(value, "shape"):
            if isinstance(value, list | tuple):
                lines.append(f"{key}: list of {len(value)}")
            continue
        arr = _np(value)
        head = f"{key}: shape {tuple(arr.shape)} {arr.dtype}"
        if key == "tokens" and arr.size == 0:
            problems.append("empty tokens")
        elif key == "tokens":
            lo, hi = int(arr.min()), int(arr.max())
            lines.append(f"{head}, ids {lo}..{hi} of vocab {vocab_size}")
            if hi >= vocab_size:
                problems.append(f"token id {hi} >= vocab {vocab_size}")
            if lo < 0:
                problems.append(f"negative token id {lo}")
        elif key == "token_lens":
            lines.append(f"{key}: {int(arr.min())}..{int(arr.max())}")
        elif key == "waveform_lens":
            sec = arr.astype(float) / sample_rate
            lines.append(f"{key}: {int(arr.min())}..{int(arr.max())} samples, "
                         f"seconds: {sec.min():.2f}..{sec.max():.2f}")
        elif np.issubdtype(arr.dtype, np.floating):
            n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
            lines.append(f"{head}, nan {n_nan}, inf {n_inf}")
            if n_nan + n_inf:
                problems.append(f"{key} has {n_nan + n_inf} NaN/inf")
        else:
            lines.append(head)
    return lines, problems


def _loader_kwargs(signature: inspect.Signature, samples: list[Any]) -> dict[str, Any]:
    """The train-loader keyword arguments this get_data_loader signature accepts."""
    wanted = {"is_eval": False, "samples": samples, "verbose": True, "assets": {},
              "num_gpus": 0}
    return {k: wanted[k] for k in _LOADER_KWARGS if k in signature.parameters}


def _vocab_size(model: Any) -> int:
    """Rows of the text embedding: a token id must be below this."""
    return int(model.text_encoder.emb.num_embeddings)


def check_loader(config_path: Path | None = None) -> bool:
    """Load one train batch the way coqui will train on it; print stats and PASS/FAIL."""
    from TTS.tts.configs.vits_config import VitsConfig  # coqui only on Colab
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits

    path = config_path or paths.REPO / train_config.CONFIG_REL
    config = VitsConfig()
    config.load_json(str(path))
    # coqui 0.28: Vits has no init_from_config; the constructor builds the tokenizer.
    model = Vits(config)
    signature = inspect.signature(model.get_data_loader)
    print(f"get_data_loader{signature}")
    train, evals = load_tts_samples(config, eval_split=True)  # 0.28 takes the whole config
    print(f"samples: {len(train)} train / {len(evals)} eval")
    loader = model.get_data_loader(config, **_loader_kwargs(signature, train))
    start = time.perf_counter()
    batch = next(iter(loader))
    try:
        import torch

        no_grad: contextlib.AbstractContextManager[Any] = torch.no_grad()
    except ImportError:  # only in tests with fake coqui; torch always comes with TTS
        no_grad = contextlib.nullcontext()
    with no_grad:
        batch = model.format_batch_on_device(model.format_batch(batch))  # adds spec and mel
    print(f"first batch (incl. worker startup + spec/mel, not a training speed) in "
          f"{time.perf_counter() - start:.2f} s")
    lines, problems = judge_batch(batch, _vocab_size(model), int(config.audio.sample_rate))
    print("\n".join(lines))
    if problems:
        print(f"FAIL: loader {'; '.join(problems)}")
        return False
    print("PASS: loader ok")
    return True
